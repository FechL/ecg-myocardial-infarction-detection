import tkinter as tk
from tkinter import ttk
import serial
import threading
import queue
import numpy as np
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ==================== KONFIGURASI ====================
SERIAL_PORT = 'COM10'          # Ganti dengan COM port Arduino-mu di Windows
BAUD_RATE = 115200
FS = 100                      
WINDOW_SIZE = 1000          
DISPLAY_WINDOW = 250  
EXPONENTIAL_FILTER_W = 0.55


# ==================== LOAD MODEL ====================
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("[+] Model dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"[-] Error memuat model: {e}")
    exit()

# ==================== FUNGSI EKSTRAKSI FITUR (ORIGINAL) ====================
def exponential_filter(signal_array, w=0.55):
    filtered = np.zeros_like(signal_array)
    filtered[0] = signal_array[0]
    for n in range(1, len(signal_array)):
        filtered[n] = w * signal_array[n] + (1 - w) * filtered[n - 1]
    return filtered

def find_qrs_peaks(ecg_signal, fs):
    # Bandpass filter 5-15 Hz
    bp_freqs = [5, 15]
    b, a = butter(3, np.array(bp_freqs) / (fs / 2), btype='bandpass')
    ecg_filtered = filtfilt(b, a, ecg_signal)
    
    # Turunan
    d_ecg = np.concatenate(([0], np.diff(ecg_filtered)))
    sq_ecg = d_ecg ** 2
    
    # Moving window integration
    win_ms = 150
    win = max(1, round((win_ms / 1000) * fs))
    mwi = np.convolve(sq_ecg, np.ones(win) / win, mode='same')
    
    # Threshold
    thr = np.mean(mwi) + 0.5 * np.std(mwi)
    min_dist = round((200 / 1000) * fs)
    
    # Find R-peaks
    locs, _ = find_peaks(mwi, height=thr, distance=min_dist)
    
    # Refinement: find exact R peaks
    search_win = round((100 / 1000) * fs)
    r_locs = np.zeros(len(locs), dtype=int)
    
    for i, loc in enumerate(locs):
        L = max(0, loc - search_win)
        R = min(len(ecg_filtered), loc + search_win + 1)
        rr = np.argmax(ecg_filtered[L:R])
        r_locs[i] = L + rr
    
    r_locs = np.unique(r_locs)
    
    # Find Q dan S peaks
    qrs_half_ms = 60
    qrs_half = round((qrs_half_ms / 1000) * fs)
    
    q_locs = np.zeros(len(r_locs), dtype=int)
    s_locs = np.zeros(len(r_locs), dtype=int)
    
    for i, r in enumerate(r_locs):
        Lq = max(0, r - qrs_half)
        Rq = r + 1
        iq = np.argmin(ecg_filtered[Lq:Rq])
        q_locs[i] = Lq + iq
        
        Ls = r
        Rs = min(len(ecg_filtered), r + qrs_half + 1)
        is_ = np.argmin(ecg_filtered[Ls:Rs])
        s_locs[i] = Ls + is_
    
    return r_locs, q_locs, s_locs, ecg_filtered

def find_t_wave(ecg_signal, fs, s_locs):
    t_locs = np.zeros(len(s_locs), dtype=int)
    t_window_ms = 400 
    t_window = round((t_window_ms / 1000) * fs)
    
    for i, s in enumerate(s_locs):
        Lt = s + round((80 / 1000) * fs) 
        Rt = min(len(ecg_signal), s + t_window)
        
        if Lt < Rt:
            it = np.argmax(ecg_signal[Lt:Rt])
            t_locs[i] = Lt + it
        else:
            t_locs[i] = s
    return t_locs

def extract_features_from_array(ecg_signal, fs=100):
    # 1. Exponential Filter
    ecg_exp = exponential_filter(ecg_signal, w=EXPONENTIAL_FILTER_W)
    
    try:
        # 2. Deteksi Peaks
        r_locs, q_locs, s_locs, ecg_bandpass = find_qrs_peaks(ecg_exp, fs)
        t_locs = find_t_wave(ecg_exp, fs, s_locs)
        
        if len(r_locs) < 2:
            return 0.0, 0.0, ecg_exp 

        # 3. Fitur 1: Q Waves
        q_values = ecg_exp[q_locs]
        q_waves = np.mean(q_values)
        
        # 4. Fitur 2: ST Elevation
        st_elevation_values = []
        for i in range(len(s_locs)):
            s_idx = s_locs[i]
            t_idx = t_locs[i]
            if s_idx < t_idx and (t_idx - s_idx) > 10:
                st_segment = ecg_exp[s_idx:t_idx]
                st_elevation_values.append(np.mean(st_segment))
        
        st_elevation = np.mean(st_elevation_values) if len(st_elevation_values) > 0 else 0.0
        
        return q_waves, st_elevation, ecg_exp
        
    except Exception as e:
        print(f"[-] Kesalahan ekstraksi: {e}")
        return 0.0, 0.0, ecg_exp

# ==================== GUI & MAIN LOGIC ====================
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Myocardial Infarction Real-Time")
        self.root.geometry("1000x600")
        
        self.data_queue = queue.Queue()
        self.ecg_buffer = []
        self.is_running = True
        
        self.setup_ui()
        self.start_serial_thread()
        self.update_gui()

    def setup_ui(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_status = ttk.Label(top_frame, text="Status: Menunggu Data...", font=("Arial", 14, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        self.lbl_q = ttk.Label(top_frame, text="Q Waves: 0.00", font=("Arial", 12))
        self.lbl_q.pack(side=tk.LEFT, padx=20)
        
        self.lbl_st = ttk.Label(top_frame, text="ST Elev: 0.00", font=("Arial", 12))
        self.lbl_st.pack(side=tk.LEFT, padx=20)

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title("Real-Time ECG Signal")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Voltage (V)")
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(0, DISPLAY_WINDOW)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def start_serial_thread(self):
        def read_from_port():
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                while self.is_running:
                    line = ser.readline().decode('utf-8').strip()
                    if line and line != "LEAD_OFF":
                        try:
                            self.data_queue.put(float(line))
                        except ValueError:
                            pass
            except Exception as e:
                print(f"[-] Kesalahan Serial: {e}")

        self.thread = threading.Thread(target=read_from_port, daemon=True)
        self.thread.start()

    def update_gui(self):
        while not self.data_queue.empty():
            self.ecg_buffer.append(self.data_queue.get())

        if len(self.ecg_buffer) > 0:
            # Hanya ambil 250 data terakhir untuk DIGAMBAR (meskipun memori menyimpan 1000)
            if len(self.ecg_buffer) > DISPLAY_WINDOW:
                plot_data = self.ecg_buffer[-DISPLAY_WINDOW:]
            else:
                plot_data = self.ecg_buffer
                
            self.line.set_data(range(len(plot_data)), plot_data)
            self.canvas.draw_idle()

        if len(self.ecg_buffer) >= WINDOW_SIZE:
            signal_window = np.array(self.ecg_buffer[:WINDOW_SIZE])
            
            q_val, st_val, _ = extract_features_from_array(signal_window)
            
            features = np.array([[q_val, st_val]])
            features_norm = scaler.transform(features)
            prediction = svm_model.predict(features_norm)[0]
            
            self.lbl_q.config(text=f"Q Waves: {q_val:.4f}")
            self.lbl_st.config(text=f"ST Elev: {st_val:.4f}")
            
            if prediction == 0:
                self.lbl_status.config(text="Diagnosis: NORMAL", foreground="green")
            else:
                self.lbl_status.config(text="Diagnosis: MI DETECTED!", foreground="red")
            
            self.ecg_buffer = self.ecg_buffer[500:]

        self.root.after(50, self.update_gui)

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()