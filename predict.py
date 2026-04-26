"""
Inference/Testing Script untuk SVM Myocardial Infarction Detector
Menggunakan trained model untuk memprediksi ECG records baru
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

# ==================== LOAD MODEL ====================
MODEL_DIR = 'models'
model_path = Path(MODEL_DIR) / 'svm_model.pkl'
scaler_path = Path(MODEL_DIR) / 'scaler.pkl'
metadata_path = Path(MODEL_DIR) / 'model_metadata.json'

# Load model files
with open(model_path, 'rb') as f:
    svm_model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print("✓ Model loaded successfully")
print(f"  Training Accuracy: {metadata['training_accuracy']*100:.2f}%")
print(f"  Feature Names: {metadata['feature_names']}")

# ==================== HELPER FUNCTIONS ====================

EXPONENTIAL_FILTER_W = metadata['exponential_filter_w']

def exponential_filter(signal, w=0.55):
    """Exponential filter dari jurnal"""
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for n in range(1, len(signal)):
        filtered[n] = w * signal[n] + (1 - w) * filtered[n - 1]
    return filtered

def find_qrs_peaks(ecg_signal, fs):
    """Deteksi Q, R, S peaks"""
    bp_freqs = [5, 15]
    b, a = butter(3, np.array(bp_freqs) / (fs / 2), btype='bandpass')
    ecg_filtered = filtfilt(b, a, ecg_signal)
    
    d_ecg = np.concatenate(([0], np.diff(ecg_filtered)))
    sq_ecg = d_ecg ** 2
    
    win_ms = 150
    win = max(1, round((win_ms / 1000) * fs))
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
    """Deteksi T wave"""
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

def extract_features(ecg_signal, fs):
    """
    Extract Q Waves dan ST Segment Elevation dari ECG signal
    
    Returns:
    - q_waves: amplitudo rata-rata Q wave
    - st_elevation: amplitudo rata-rata ST segment
    - status: 'OK' atau error message
    """
    try:
        # Apply exponential filter
        ecg_filtered = exponential_filter(ecg_signal, w=EXPONENTIAL_FILTER_W)
        
        # Deteksi peaks
        r_locs, q_locs, s_locs, ecg_bandpass = find_qrs_peaks(ecg_filtered, fs)
        
        # Deteksi T wave
        t_locs = find_t_wave(ecg_filtered, fs, s_locs)
        
        if len(r_locs) < 2:
            return None, None, 'ERROR_NO_PEAKS'
        
        # FITUR 1: Pathological Q Waves
        q_values = ecg_filtered[q_locs]
        q_waves = np.mean(q_values)
        
        # FITUR 2: ST Segment Elevation
        st_elevation_values = []
        for i in range(len(s_locs)):
            s_idx = s_locs[i]
            t_idx = t_locs[i]
            if s_idx < t_idx and (t_idx - s_idx) > 10:
                st_segment = ecg_filtered[s_idx:t_idx]
                st_elevation_values.append(np.mean(st_segment))
        
        if len(st_elevation_values) == 0:
            return None, None, 'ERROR_NO_ST'
        
        st_elevation = np.mean(st_elevation_values)
        
        return q_waves, st_elevation, 'OK'
        
    except Exception as e:
        return None, None, f'ERROR_{str(e)}'

# ==================== PREDICTION FUNCTION ====================

def predict_ecg(ecg_signal, fs, return_confidence=True):
    """
    Predict apakah ECG adalah Normal atau Myocardial Infarction
    
    Args:
    - ecg_signal: numpy array dari ECG signal (single lead)
    - fs: sampling frequency (Hz)
    - return_confidence: jika True, return juga confidence score
    
    Returns:
    - prediction: 'NORMAL' atau 'MYOCARDIAL INFARCTION'
    - confidence: float 0-100 (jika return_confidence=True)
    - features: dict dengan Q_Waves dan ST_Elevation values
    - status: 'OK' atau error message
    """
    
    # Extract features
    q_waves, st_elevation, status = extract_features(ecg_signal, fs)
    
    if status != 'OK':
        return 'ERROR', None, None, status
    
    # Normalize dengan scaler
    features = np.array([[q_waves, st_elevation]])
    features_normalized = scaler.transform(features)
    
    # Predict
    prediction_code = svm_model.predict(features_normalized)[0]
    
    # Confidence (decision function)
    decision = svm_model.decision_function(features_normalized)[0]
    confidence = abs(decision) * 100  # Convert to percentage
    
    # Map code to label
    prediction_label = 'NORMAL' if prediction_code == 0 else 'MYOCARDIAL INFARCTION'
    
    features_dict = {
        'Q_Waves': float(q_waves),
        'ST_Elevation': float(st_elevation),
        'Q_Waves_normalized': float(features_normalized[0, 0]),
        'ST_Elevation_normalized': float(features_normalized[0, 1])
    }
    
    if return_confidence:
        return prediction_label, confidence, features_dict, 'OK'
    else:
        return prediction_label, None, features_dict, 'OK'

# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    import wfdb
    
    print("\n" + "="*70)
    print("TESTING SVM MODEL ON NEW ECG DATA")
    print("="*70)
    
    # Load test record
    test_record_path = 'dataset/records100/00000/00001_lr'
    
    try:
        record = wfdb.rdrecord(test_record_path)
        lead_ii = record.p_signal[:, 1]  # Lead II
        fs = record.fs
        
        print(f"\n[*] Testing record: {test_record_path}")
        print(f"    Sampling rate: {fs} Hz")
        print(f"    Signal length: {len(lead_ii)} samples")
        
        # Predict
        prediction, confidence, features, status = predict_ecg(lead_ii, fs)
        
        if status == 'OK':
            print(f"\n[RESULT]")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"\n[EXTRACTED FEATURES]")
            print(f"  Q_Waves: {features['Q_Waves']:.6f}")
            print(f"  ST_Elevation: {features['ST_Elevation']:.6f}")
            print(f"\n[NORMALIZED FEATURES]")
            print(f"  Q_Waves (normalized): {features['Q_Waves_normalized']:.4f}")
            print(f"  ST_Elevation (normalized): {features['ST_Elevation_normalized']:.4f}")
        else:
            print(f"✗ Error: {status}")
        
    except Exception as e:
        print(f"✗ Error loading record: {e}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
