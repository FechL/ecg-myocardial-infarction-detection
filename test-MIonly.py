import numpy as np
import pickle
import csv
import os
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime
import matplotlib.pyplot as plt
import ast

# ==================== KONFIGURASI ====================
DATASET_PATH = './dataset/'
FS = 100  # Sampling rate untuk records100
WINDOW_MI = 1000  # 10 detik untuk SVM MI
EXPONENTIAL_FILTER_W = 0.55

# Load database untuk menemukan MI subjects
db = pd.read_csv(os.path.join(DATASET_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
db['scp_codes'] = db['scp_codes'].apply(lambda x: ast.literal_eval(x))

# Identifikasi MI subjects
def has_mi_diagnosis(scp_codes):
    """Cek apakah subject memiliki diagnosa MI"""
    mi_codes = ['IMI', 'ILMI', 'AMI', 'ALMI', 'ASMI']
    return any(code in scp_codes for code in mi_codes)

# Gunakan 10 random MI subjects
mi_subjects = db[db['scp_codes'].apply(has_mi_diagnosis)]
np.random.seed(42)
random_mi_ids = np.random.choice(mi_subjects.index, size=min(2, len(mi_subjects)), replace=False)
TEST_SUBJECTS = []

for ecg_id in random_mi_ids:
    filename = db.loc[ecg_id, 'filename_lr']
    scp_codes = db.loc[ecg_id, 'scp_codes']
    mi_type = ', '.join([k for k in scp_codes.keys() if k in ['IMI', 'ILMI', 'AMI', 'ALMI', 'ASMI']])
    TEST_SUBJECTS.append({
        'ecg_id': ecg_id,
        'filename': filename,
        'diagnosis': mi_type
    })

print(f"[+] {len(TEST_SUBJECTS)} random MI subjects loaded")

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
    """Exponential filter untuk ECG smoothing"""
    filtered = np.zeros_like(signal_array)
    filtered[0] = signal_array[0]
    for n in range(1, len(signal_array)):
        filtered[n] = w * signal_array[n] + (1 - w) * filtered[n - 1]
    return filtered

def find_qrs_peaks(ecg_signal, fs):
    """Menemukan lokasi Q, R, S peaks dari ECG signal"""
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
    """Menemukan lokasi T wave dari ECG signal"""
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
    """Ekstraksi fitur MI dari ECG signal"""
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
    except Exception as e:
        print(f"[-] Error ekstraksi fitur: {e}")
        return 0.0, 0.0

def load_ecg_signal(ecg_file):
    """Load ECG signal dari file wfdb"""
    try:
        signal, meta = wfdb.rdsamp(ecg_file)
        return signal, meta
    except Exception as e:
        print(f"[-] Error membaca file ECG {ecg_file}: {e}")
        return None, None

def test_mi_subjects():
    """Test MI detection pada multiple subjects"""
    
    # Setup CSV output - save to results folder
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_filename = os.path.join(results_dir, f"MI_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['ECG_ID', 'Diagnosis_Label', 'Window_Number', 'Q_Waves', 'ST_Elevation', 
                         'Prediction', 'Confidence', 'Status', 'Lead'])
    csv_file.flush()
    
    print("="*80)
    print("ECG MI DETECTION TEST - PhysioNet Dataset (Gui.py Methodology)")
    print("="*80)
    
    results_summary = []
    
    for idx, subject in enumerate(TEST_SUBJECTS, 1):
        ecg_id = subject['ecg_id']
        filename = subject['filename']
        diagnosis_label = subject['diagnosis']
        
        print(f"[{idx}/{len(TEST_SUBJECTS)}] ECG {ecg_id}: {diagnosis_label[:15]:<15}", end=" | ")
        
        # Load ECG signal
        ecg_file = os.path.join(DATASET_PATH, filename)
        ecg_signal, meta = load_ecg_signal(ecg_file)
        
        if ecg_signal is None:
            print("FAILED")
            continue
        
        # Extract dari Lead II
        ecg_lead_ii = ecg_signal[:, 1]  # Lead II
        
        # Single lead analysis
        q_val, st_val = extract_mi_features(ecg_lead_ii, fs=FS)
        
        # Predict
        features = np.array([[q_val, st_val]])
        features_norm = scaler.transform(features)
        prediction = svm_model.predict(features_norm)[0]
        
        # Confidence score
        decision_func = svm_model.decision_function(features_norm)[0]
        confidence = abs(decision_func)
        
        pred_label = "NORMAL" if prediction == 0 else "MI"
        
        # Write to CSV
        csv_writer.writerow([ecg_id, diagnosis_label, 0, f"{q_val:.6f}", f"{st_val:.6f}", 
                           prediction, f"{confidence:.6f}", pred_label, "II"])
        csv_file.flush()
        
        # Accuracy = 100% jika terdeteksi MI, 0% jika tidak (untuk MI subjects)
        accuracy = 100.0 if prediction == 1 else 0.0
        
        print(f"Q={q_val:.4f} | ST={st_val:.4f} | Pred: {pred_label:6s} | Accuracy: {accuracy:.1f}% | Conf: {confidence:.3f}")
        
        results_summary.append({
            'ECG_ID': ecg_id,
            'Diagnosis': diagnosis_label,
            'Prediction': pred_label,
            'Accuracy': f"{accuracy:.1f}%",
            'Confidence': f"{confidence:.4f}"
        })
    
    # Close CSV
    csv_file.close()
    
    # Calculate overall accuracy
    total_subjects = len(results_summary)
    mi_detections = sum(1 for r in results_summary if r['Prediction'] == 'MI')
    overall_accuracy = (mi_detections / total_subjects * 100) if total_subjects > 0 else 0
    overall_conf = np.mean([float(r['Confidence']) for r in results_summary]) if results_summary else 0
    
    print("\n" + "="*80)
    print(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({mi_detections}/{total_subjects} subjects detected as MI)")
    print(f"Average Confidence: {overall_conf:.4f}")
    print(f"Results saved: {csv_filename}")
    print("="*80)

if __name__ == "__main__":
    test_mi_subjects()
    print("\n[+] Testing completed!")

