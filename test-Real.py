import tkinter as tk
from tkinter import ttk
import serial
import threading
import queue
import numpy as np
import pickle
import csv
import os
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ==================== KONFIGURASI ====================
SERIAL_PORT = 'COM10'         # Sesuaikan port Anda
BAUD_RATE = 115200
FS = 100                      
WINDOW_MI = 1000              # 10 detik untuk SVM MI
WINDOW_BPM = 1500             # 15 detik untuk BPM
DISPLAY_WINDOW = 300          # Tampilkan 3 detik di grafik agar "Ter-zoom"
EXPONENTIAL_FILTER_W = 0.55   

# ==================== LOAD MODEL SVM ====================
try:
    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('./models/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("[+] Model SVM dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"[-] Error memuat model: {e}")
    exit()

# ==================== FUNGSI PEMROSESAN SINYAL ====================
def exponential_filter(signal_array, w=0.55):
    filtered = np.zeros_like(signal_array)
    filtered[0] = signal_array[0]
    for n in range(1, len(signal_array)):
        filtered[n] = w * signal_array[n] + (1 - w) * filtered[n - 1]
    return filtered

def find_qrs_peaks(ecg_signal, fs):
    bp_freqs = [5, 15]
    b, a = butter(3, np.array(bp_freqs) / (fs / 2), btype='bandpass')
    ecg_filtered = filtfilt(b, a, ecg_signal)
    
    d_ecg = np.concatenate(([0], np.diff(ecg_filtered)))
    sq_ecg = d_ecg ** 2
    
    win = max(1, round((150 / 1000) * fs))
    mwi = np.convolve(sq_ecg, np.ones(win) / win, mode='same')
    
    thr = np.mean(mwi) + 0.5 * np.std(mwi)
    min_dist = round((200 / 1000) * fs)
    locs, _ = find_peaks(mwi, height=thr, distance=min_dist)
    
    search_win = round((100 / 1000) * fs)
    r_locs = np.zeros(len(locs), dtype=int)
    for i, loc in enumerate(locs):
        L = max(0, loc - search_win)
        R = min(len(ecg_filtered), loc + search_win + 1)
        rr = np.argmax(ecg_filtered[L:R])
        r_locs[i] = L + rr
    r_locs = np.unique(r_locs)
    
    qrs_half = round((60 / 1000) * fs)
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
    t_window = round((400 / 1000) * fs)
    for i, s in enumerate(s_locs):
        Lt = s + round((80 / 1000) * fs) 
        Rt = min(len(ecg_signal), s + t_window)
        if Lt < Rt:
            it = np.argmax(ecg_signal[Lt:Rt])
            t_locs[i] = Lt + it
        else:
            t_locs[i] = s
    return t_locs

def extract_mi_features(ecg_signal, fs=100):
    ecg_exp = exponential_filter(ecg_signal, w=EXPONENTIAL_FILTER_W)
    try:
        r_locs, q_locs, s_locs, _ = find_qrs_peaks(ecg_exp, fs)
        t_locs = find_t_wave(ecg_exp, fs, s_locs)
        
        if len(r_locs) < 2: 
            return 0.0, 0.0
            
        q_values = ecg_exp[q_locs]
        q_waves = np.mean(q_values)
        
        st_elevation_values = []
        for i in range(len(s_locs)):
            s_idx = s_locs[i]
            t_idx = t_locs[i]
            if s_idx < t_idx and (t_idx - s_idx) > 10:
                st_segment = ecg_exp[s_idx:t_idx]
                st_elevation_values.append(np.mean(st_segment))
        st_elevation = np.mean(st_elevation_values) if len(st_elevation_values) > 0 else 0.0
        
        return q_waves, st_elevation
    except Exception:
        return 0.0, 0.0

def count_r_peaks_for_bpm(ecg_signal, fs=100):
    """Menghitung jumlah puncak R untuk BPM"""
    bp_freqs = [5, 15]
    b, a = butter(3, np.array(bp_freqs) / (fs / 2), btype='bandpass')
    ecg_filtered = filtfilt(b, a, ecg_signal)
    
    d_ecg = np.concatenate(([0], np.diff(ecg_filtered)))
    sq_ecg = d_ecg ** 2
    
    win = max(1, round((150 / 1000) * fs))
    mwi = np.convolve(sq_ecg, np.ones(win) / win, mode='same')
    
    thr = np.mean(mwi) + 0.5 * np.std(mwi)
    min_dist = round((200 / 1000) * fs)
    locs, _ = find_peaks(mwi, height=thr, distance=min_dist)
    
    return len(locs)

# ==================== GUI MAIN LOGIC ====================
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi MI & BPM Real-Time")
        self.root.geometry("1000x650")
        
        self.data_queue = queue.Queue()
        self.ecg_buffer = []
        
        self.total_samples_received = 0 # Untuk menghitung detik
        self.new_samples_count = 0      # Untuk trigger prediksi (tiap 100 sampel)
        self.is_running = True
        self.is_recording = False
        self.recording_start_time = None
        
        # ===== CSV LOGGING SETUP =====
        self.results_dir = './results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.csv_filename = os.path.join(self.results_dir, f"ecg_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Q_Waves', 'ST_Elevation', 'Label', 'Diagnosis'])
        self.csv_file.flush()
        self.last_csv_save_time = 0
        print(f"[+] CSV logging dimulai: {self.csv_filename}")
        
        self.setup_ui()
        self.start_serial_thread()
        self.update_gui()

    def setup_ui(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 1. Baris Pertama: Timer / Waktu Perekaman
        self.lbl_timer = tk.Label(top_frame, text="Waktu Perekaman: 00:00", font=("Arial", 16, "bold"), fg="#2980b9")
        self.lbl_timer.pack(pady=(0, 10))
        
        # Tombol START/STOP Recording
        button_frame = tk.Frame(top_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.btn_start = tk.Button(button_frame, text="START RECORDING (15s)", command=self.start_recording,
                                   bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=25)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(button_frame, text="STOP", command=self.stop_recording,
                                  bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), width=10, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        self.lbl_recording_status = tk.Label(button_frame, text="Status: SIAP", font=("Arial", 11, "bold"), fg="#27ae60")
        self.lbl_recording_status.pack(side=tk.LEFT, padx=20)

        # 2. Baris Kedua: Status MI & Fitur
        frame_mi = tk.Frame(top_frame)
        frame_mi.pack(fill=tk.X, pady=5)
        
        self.lbl_mi_status = tk.Label(frame_mi, text="Diagnosis: Menunggu (10s)...", font=("Arial", 14, "bold"))
        self.lbl_mi_status.pack(side=tk.LEFT, padx=20)
        
        self.lbl_q = tk.Label(frame_mi, text="Q Waves: 0.00", font=("Arial", 12))
        self.lbl_q.pack(side=tk.LEFT, padx=20)
        
        self.lbl_st = tk.Label(frame_mi, text="ST Elev: 0.00", font=("Arial", 12))
        self.lbl_st.pack(side=tk.LEFT, padx=20)

        # 3. Baris Ketiga: Status BPM
        frame_bpm = tk.Frame(top_frame)
        frame_bpm.pack(fill=tk.X, pady=5)
        
        self.lbl_bpm_status = tk.Label(frame_bpm, text="Heart Rate: -- BPM (Menunggu 15s...)", font=("Arial", 14, "bold"), fg="#e67e22")
        self.lbl_bpm_status.pack(side=tk.LEFT, padx=20)

        # 4. Baris Keempat: Grafik Matplotlib (Ter-zoom 3 detik)
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_title("Live ECG Signal")
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
            ser = None
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                while self.is_running:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    if line == "LEAD_OFF":
                        self.data_queue.put(-999.0) # Kode sinyal lepas
                        continue
                    try:
                        self.data_queue.put(float(line))
                    except ValueError:
                        pass
            except Exception as e:
                print(f"[-] Kesalahan Serial: {e}")
            finally:
                if ser is not None and ser.is_open:
                    ser.close()

        self.thread = threading.Thread(target=read_from_port, daemon=True)
        self.thread.start()

    def start_recording(self):
        """Mulai recording selama 15 detik"""
        self.is_recording = True
        self.recording_start_time = datetime.now()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_recording_status.config(text="Status: RECORDING (15s)", fg="#e74c3c")
        print("[+] Recording dimulai! (Durasi: 15 detik)")

    def stop_recording(self):
        """Hentikan recording"""
        self.is_recording = False
        self.recording_start_time = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_recording_status.config(text="Status: SIAP", fg="#27ae60")
        print("[-] Recording dihentikan.")

    def update_gui(self):
        while not self.data_queue.empty():
            val = self.data_queue.get()
            
            # Jika sensor lepas (LEAD_OFF)
            if val == -999.0:
                self.lbl_timer.config(text="LEAD OFF - Pasang Sensor!", fg="red")
                self.lbl_mi_status.config(text="Diagnosis: Sensor Terlepas", fg="red")
                self.lbl_bpm_status.config(text="Heart Rate: Sensor Terlepas", fg="red")
                self.ecg_buffer.clear()
                self.total_samples_received = 0 # Reset waktu
                self.new_samples_count = 0
                continue

            # Jika data valid masuk
            self.ecg_buffer.append(val)
            self.total_samples_received += 1
            self.new_samples_count += 1

            # Batasi memori maksimal 15 detik (1500 sampel)
            if len(self.ecg_buffer) > WINDOW_BPM:
                self.ecg_buffer.pop(0)

        # 1. Update Timer (Detik/Menit) secara real-time
        if self.total_samples_received > 0:
            total_detik = self.total_samples_received // FS
            menit = total_detik // 60
            detik = total_detik % 60
            self.lbl_timer.config(text=f"Waktu Perekaman: {menit:02d}:{detik:02d}", fg="#2980b9")

        # 2. Update Plot secara real-time (tampilkan 300 sampel terakhir agar "zoom")
        if len(self.ecg_buffer) > 0:
            plot_data = self.ecg_buffer[-DISPLAY_WINDOW:] if len(self.ecg_buffer) > DISPLAY_WINDOW else self.ecg_buffer
            self.line.set_data(range(len(plot_data)), plot_data)
            self.canvas.draw_idle()

        # 3. Cek apakah recording sudah 15 detik, jika ya auto-stop
        if self.is_recording and self.recording_start_time is not None:
            elapsed_time = (datetime.now() - self.recording_start_time).total_seconds()
            self.lbl_recording_status.config(text=f"Status: RECORDING ({15 - int(elapsed_time):02d}s)")
            if elapsed_time >= 15:
                self.stop_recording()
                print("[+] Recording otomatis berhenti (15 detik tercapai).")
        
        # 4. Prediksi AI & BPM (Di-trigger tiap 1 detik / 100 sampel data baru)
        if self.new_samples_count >= 100:
            self.new_samples_count = 0 # Reset trigger
            signal_window = np.array(self.ecg_buffer)
            
            # --- DETEKSI MI (Butuh memori 10 detik terakhir) ---
            if len(signal_window) >= WINDOW_MI:
                # Ambil 1000 sampel terakhir saja untuk SVM
                signal_10s = signal_window[-WINDOW_MI:]
                q_val, st_val = extract_mi_features(signal_10s, fs=FS)
                
                features = np.array([[q_val, st_val]])
                features_norm = scaler.transform(features)
                prediction = svm_model.predict(features_norm)[0]
                
                self.lbl_q.config(text=f"Q Waves: {q_val:.4f}")
                self.lbl_st.config(text=f"ST Elev: {st_val:.4f}")
                
                diagnosis = "NORMAL" if prediction == 0 else "MI DETECTED!"
                label_text = "NORMAL" if prediction == 0 else "MI"
                
                if prediction == 0:
                    self.lbl_mi_status.config(text="Diagnosis: NORMAL", fg="green")
                else:
                    self.lbl_mi_status.config(text="Diagnosis: MI DETECTED!", fg="red")
                
                # ===== SIMPAN KE CSV HANYA SAAT RECORDING =====
                if self.is_recording:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    self.csv_writer.writerow([timestamp, f"{q_val:.6f}", f"{st_val:.6f}", label_text, diagnosis])
                    self.csv_file.flush()
                    print(f"[*] Data disimpan ke CSV - Q: {q_val:.4f}, ST: {st_val:.4f}, Label: {label_text}")

            # --- DETEKSI BPM (Butuh memori 15 detik penuh) ---
            if len(signal_window) >= WINDOW_BPM:
                beat_count = count_r_peaks_for_bpm(signal_window, fs=FS)
                bpm_val = beat_count * 4 # 15 detik x 4 = 1 menit
                
                # Menampilkan nilai BPM saja tanpa klasifikasi (selalu warna orange/default)
                self.lbl_bpm_status.config(text=f"Heart Rate: {bpm_val} BPM", fg="#e67e22")

        # Loop setiap 50ms
        self.root.after(50, self.update_gui)

    def on_closing(self):
        self.is_running = False
        # ===== TUTUP CSV FILE =====
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            print(f"[+] CSV file ditutup: {self.csv_filename}")
        self.root.after(100, self._destroy_app)

    def _destroy_app(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()