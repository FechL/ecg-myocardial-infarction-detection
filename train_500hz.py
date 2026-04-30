"""
Training SVM Model untuk Deteksi Myocardial Infarction - 500 Hz Version
Berdasarkan Jurnal: "Sistem Deteksi Myocardial Infarction Berdasarkan 
Pathological Q Waves dan ST Segment Elevation Menggunakan Support Vector Machine"

Features:
- Q Waves: nilai amplitudo gelombang Q
- ST Segment Elevation: rata-rata amplitudo antara gelombang S dan T

Klasifikasi: Normal vs Myocardial Infarction (MI)

PERBEDAAN DARI train.py:
- Sampling rate: 500 Hz (default PTB-XL)
- Samples per window: 200 (2 seconds of data)
- Untuk ESP32 dengan memory lebih besar
"""

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
DATASET_PATH = 'dataset'
DB_PATH = 'dataset/ptbxl_database.csv'
OUTPUT_DIR = 'models'
OUTPUT_SUFFIX = '_500hz'  # Suffix untuk 500 Hz model

# Buat output directory jika belum ada
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Hyperparameter
EXPONENTIAL_FILTER_W = 0.55  # dari jurnal
MIN_HR = 40  # BPM
MAX_HR = 200  # BPM
SAMPLING_RATE = 500  # Hz (PTB-XL native rate)
SAMPLES_PER_WINDOW = 200  # 0.4 seconds (fixed window size for ESP32)

print("="*60)
print("TRAINING SVM UNTUK DETEKSI MYOCARDIAL INFARCTION - 500 Hz")
print(f"Sampling Rate: {SAMPLING_RATE} Hz")
print(f"Samples per Window: {SAMPLES_PER_WINDOW} ({SAMPLES_PER_WINDOW/SAMPLING_RATE:.1f} seconds)")
print("="*60)

# ==================== 1. LOAD DATABASE ====================
print("\n[1] Loading database metadata...")
try:
    db_df = pd.read_csv(DB_PATH)
    print(f"  Database loaded: {len(db_df)} records")
except Exception as e:
    print(f"  Error loading database: {e}")
    exit(1)

# ==================== 2. FILTER DATA BERDASARKAN DIAGNOSIS ====================
print("\n[2] Filtering data by diagnosis (NORM vs MI)...")

def get_primary_diagnosis(scp_codes_str):
    """
    Extract primary diagnosis dari scp_codes
    NORM: "{'NORM': 100.0, 'SR': 0.0}"
    MI: "{'IMI': 35.0, 'ABQRS': 0.0}"
    """
    try:
        if pd.isna(scp_codes_str):
            return None
        
        # Parse dictionary string
        import ast
        codes_dict = ast.literal_eval(scp_codes_str)
        
        # Diagnosis prioritas
        mi_codes = ['IMI', 'AMI', 'ASMI', 'LMI', 'ALMI', 'ILMI', 'IPLMI', 'APICAL']
        norm_codes = ['NORM']
        
        # Cek MI dulu
        for code in codes_dict.keys():
            if code in mi_codes:
                return 'MI'
        
        # Cek NORM
        for code in codes_dict.keys():
            if code in norm_codes:
                return 'NORM'
        
        return None
    except:
        return None

# Add diagnosis column
db_df['diagnosis'] = db_df['scp_codes'].apply(get_primary_diagnosis)

# Filter hanya NORM dan MI
norm_df = db_df[db_df['diagnosis'] == 'NORM'].copy()
mi_df = db_df[db_df['diagnosis'] == 'MI'].copy()

print(f"  NORM records: {len(norm_df)}")
print(f"  MI records: {len(mi_df)}")

# ==================== 3. EKSTRAKSI FITUR ECG ====================
print("\n[3] Extracting ECG features...")

def exponential_filter(signal, w=0.55):
    """
    Exponential filter dari jurnal
    Equation: Y_n = w * X_n + (1 - w) * Y_{n-1}
    """
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    
    for n in range(1, len(signal)):
        filtered[n] = w * signal[n] + (1 - w) * filtered[n - 1]
    
    return filtered

def find_qrs_peaks(ecg_signal, fs):
    """
    Deteksi Q, R, S peaks menggunakan modified Pan-Tompkins
    """
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
        # Q: minimum sebelum R
        Lq = max(0, r - qrs_half)
        Rq = r + 1
        iq = np.argmin(ecg_filtered[Lq:Rq])
        q_locs[i] = Lq + iq
        
        # S: minimum setelah R
        Ls = r
        Rs = min(len(ecg_filtered), r + qrs_half + 1)
        is_ = np.argmin(ecg_filtered[Ls:Rs])
        s_locs[i] = Ls + is_
    
    return r_locs, q_locs, s_locs, ecg_filtered

def find_t_wave(ecg_signal, fs, s_locs):
    """
    Deteksi T wave - puncak setelah S wave
    """
    t_locs = np.zeros(len(s_locs), dtype=int)
    t_window_ms = 400  # window untuk cari T wave
    t_window = round((t_window_ms / 1000) * fs)
    
    for i, s in enumerate(s_locs):
        Lt = s + round((80 / 1000) * fs)  # start 80ms setelah S
        Rt = min(len(ecg_signal), s + t_window)
        
        if Lt < Rt:
            it = np.argmax(ecg_signal[Lt:Rt])
            t_locs[i] = Lt + it
        else:
            t_locs[i] = s
    
    return t_locs

def extract_features(record_path, fs_target=500, samples_window=200):
    """
    Extract Q Waves dan ST Segment Elevation dari record
    Resample ke target fs jika diperlukan
    
    Returns:
    - q_waves: nilai amplitudo Q (pathological)
    - st_elevation: rata-rata amplitudo S-T
    - status: 'OK' atau 'ERROR' jika ada masalah
    """
    try:
        # Load record
        record = wfdb.rdrecord(record_path)
        
        # Gunakan lead II (index 1) - yang paling informatif untuk deteksi Q waves
        if record.p_signal.shape[1] < 2:
            return None, None, 'ERROR_CHANNELS'
        
        lead_ii = record.p_signal[:, 1]  # Lead II
        fs = record.fs
        
        # Resample jika fs tidak sesuai
        if fs != fs_target:
            # Resample ke fs_target
            from scipy.interpolate import interp1d
            ratio = fs_target / fs
            n_samples_new = int(len(lead_ii) * ratio)
            t_old = np.linspace(0, 1, len(lead_ii))
            t_new = np.linspace(0, 1, n_samples_new)
            f = interp1d(t_old, lead_ii, kind='cubic')
            lead_ii = f(t_new)
            fs = fs_target
        
        # Apply exponential filter
        ecg_filtered = exponential_filter(lead_ii, w=EXPONENTIAL_FILTER_W)
        
        # Deteksi peaks
        r_locs, q_locs, s_locs, ecg_bandpass = find_qrs_peaks(ecg_filtered, fs)
        
        # Deteksi T wave
        t_locs = find_t_wave(ecg_filtered, fs, s_locs)
        
        if len(r_locs) < 2:
            return None, None, 'ERROR_NO_PEAKS'
        
        # ============================================
        # FITUR 1: Pathological Q Waves
        # ============================================
        # Nilai rata-rata amplitudo gelombang Q
        q_values = ecg_filtered[q_locs]
        q_waves = np.mean(q_values)  # Rata-rata Q amplitude
        
        # ============================================
        # FITUR 2: ST Segment Elevation
        # ============================================
        # Rata-rata amplitudo antara gelombang S dan T
        st_elevation_values = []
        
        for i in range(len(s_locs)):
            s_idx = s_locs[i]
            t_idx = t_locs[i]
            
            if s_idx < t_idx and (t_idx - s_idx) > 10:  # minimal jarak
                # Rata-rata nilai antara S dan T
                st_segment = ecg_filtered[s_idx:t_idx]
                st_elevation_values.append(np.mean(st_segment))
        
        if len(st_elevation_values) == 0:
            return None, None, 'ERROR_NO_ST'
        
        st_elevation = np.mean(st_elevation_values)
        
        return q_waves, st_elevation, 'OK'
        
    except Exception as e:
        return None, None, f'ERROR_{str(e)}'

# ==================== 4. EXTRACT DATA LATIH ====================
print("\n[4] Extracting training data features (500 Hz)...")

# Ambil balanced sample: 18 NORM + 18 MI = 36 total (sesuai jurnal)
n_samples_per_class = 18

norm_sample = norm_df.sample(n=min(n_samples_per_class, len(norm_df)), random_state=42)
mi_sample = mi_df.sample(n=min(n_samples_per_class, len(mi_df)), random_state=42)

print(f"  NORM samples: {len(norm_sample)}")
print(f"  MI samples: {len(mi_sample)}")

# Extract features
X_train = []
y_train = []
feature_names = ['Q_Waves', 'ST_Elevation']

print("  Extracting NORM features...")
for idx, row in norm_sample.iterrows():
    record_path = Path(DATASET_PATH) / row['filename_lr']
    q_waves, st_elev, status = extract_features(str(record_path), fs_target=SAMPLING_RATE, 
                                                samples_window=SAMPLES_PER_WINDOW)
    
    if status == 'OK':
        X_train.append([q_waves, st_elev])
        y_train.append(0)  # 0 = NORM
        print(f"    ECG ID {row['ecg_id']}: Q={q_waves:.4f}, ST={st_elev:.4f}")
    else:
        print(f"    ECG ID {row['ecg_id']}: {status}")

print("  Extracting MI features...")
for idx, row in mi_sample.iterrows():
    record_path = Path(DATASET_PATH) / row['filename_lr']
    q_waves, st_elev, status = extract_features(str(record_path), fs_target=SAMPLING_RATE,
                                                samples_window=SAMPLES_PER_WINDOW)
    
    if status == 'OK':
        X_train.append([q_waves, st_elev])
        y_train.append(1)  # 1 = MI
        print(f"    ECG ID {row['ecg_id']}: Q={q_waves:.4f}, ST={st_elev:.4f}")
    else:
        print(f"    ECG ID {row['ecg_id']}: {status}")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"  Total training samples extracted: {len(X_train)}")
print(f"    NORM: {np.sum(y_train == 0)}")
print(f"    MI: {np.sum(y_train == 1)}")

# ==================== 5. NORMALISASI Z-SCORE ====================
print("\n[5] Normalizing features with Z-Score...")

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)

print(f"  Mean: {scaler.mean_}")
print(f"  Std: {scaler.scale_}")

# ==================== 6. TRAIN SVM ====================
print("\n[6] Training SVM classifier...")

# SVM dengan kernel RBF (default)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_normalized, y_train)

print(f"  Number of support vectors: {len(svm_model.support_vectors_)}")

# Count support vectors per class
sv_indices = svm_model.support_
sv_labels = y_train[sv_indices]
print(f"  Class 0 (NORM) support vectors: {np.sum(sv_labels == 0)}")
print(f"  Class 1 (MI) support vectors: {np.sum(sv_labels == 1)}")

# ==================== 7. EVALUASI TRAINING ====================
print("\n[7] Evaluating training model...")

y_pred_train = svm_model.predict(X_train_normalized)
accuracy_train = accuracy_score(y_train, y_pred_train)

print(f"  Training Accuracy: {accuracy_train * 100:.2f}%")
print("  Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))
print("  Classification Report:")
print(classification_report(y_train, y_pred_train, target_names=['NORM', 'MI']))

# ==================== 8. EXTRACT WEIGHTS ====================
print("\n[8] Extracting SVM weights...")

# Untuk SVM dengan RBF kernel, weights tidak langsung tersedia
# Kita bisa menggunakan coefficients dan support vectors
coefficients = svm_model.dual_coef_[0]
support_vectors = svm_model.support_vectors_
intercept = svm_model.intercept_[0]

print(f"  Intercept (bias): {intercept:.6f}")
print(f"  Number of coefficients: {len(coefficients)}")

# Untuk feature importance, kita bisa hitung menggunakan permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(svm_model, X_train_normalized, y_train, 
                               n_repeats=10, random_state=42)

print("  Feature Importance (Permutation):")
for i, name in enumerate(feature_names):
    print(f"  {name}: {result.importances_mean[i]:.6f}")

# ==================== 9. SIMPAN MODEL ====================
print("\n[9] Saving model files...")

# Simpan SVM model
model_path = Path(OUTPUT_DIR) / f'svm_model{OUTPUT_SUFFIX}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(svm_model, f)

# Simpan scaler
scaler_path = Path(OUTPUT_DIR) / f'scaler{OUTPUT_SUFFIX}.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Simpan metadata
metadata = {
    'feature_names': feature_names,
    'sampling_rate': SAMPLING_RATE,
    'samples_per_window': SAMPLES_PER_WINDOW,
    'window_duration_seconds': SAMPLES_PER_WINDOW / SAMPLING_RATE,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_std': scaler.scale_.tolist(),
    'svm_intercept': float(intercept),
    'training_accuracy': float(accuracy_train),
    'n_training_samples': len(X_train),
    'n_support_vectors': len(support_vectors),
    'kernel': 'rbf',
    'exponential_filter_w': EXPONENTIAL_FILTER_W,
    'feature_importance': result.importances_mean.tolist()
}

metadata_path = Path(OUTPUT_DIR) / f'model_metadata{OUTPUT_SUFFIX}.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# Simpan training data untuk reference
training_data = {
    'X_train': X_train.tolist(),
    'y_train': y_train.tolist(),
    'X_train_normalized': X_train_normalized.tolist(),
    'feature_names': feature_names
}

training_path = Path(OUTPUT_DIR) / f'training_data{OUTPUT_SUFFIX}.json'
with open(training_path, 'w') as f:
    json.dump(training_data, f, indent=2)

# ==================== 10. SUMMARY ====================
print("\nTRAINING SUMMARY (500 Hz VERSION)")
print(f"  Model Type: SVM with RBF kernel")
print(f"  Sampling Rate: {SAMPLING_RATE} Hz")
print(f"  Samples per Window: {SAMPLES_PER_WINDOW} ({SAMPLES_PER_WINDOW/SAMPLING_RATE:.1f} seconds)")
print(f"  Training Samples: {len(X_train)}")
print(f"    - NORM: {np.sum(y_train == 0)}")
print(f"    - MI: {np.sum(y_train == 1)}")
print(f"  Features: {', '.join(feature_names)}")
print(f"  Training Accuracy: {accuracy_train * 100:.2f}%")
print(f"  Support Vectors: {len(support_vectors)}")
print(f"  Weights (Permutation Importance):")
print(f"    - Q Waves: {result.importances_mean[0]:.6f}")
print(f"    - ST Elevation: {result.importances_mean[1]:.6f}")

# ==================== 11. VISUALISASI ====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Feature distribution
ax = axes[0, 0]
norm_data = X_train[y_train == 0]
mi_data = X_train[y_train == 1]

ax.scatter(norm_data[:, 0], norm_data[:, 1], label='NORM', alpha=0.6, s=100)
ax.scatter(mi_data[:, 0], mi_data[:, 1], label='MI', alpha=0.6, s=100)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_title('Feature Distribution (Before Normalization) - 500 Hz')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Normalized feature distribution
ax = axes[0, 1]
norm_data_norm = X_train_normalized[y_train == 0]
mi_data_norm = X_train_normalized[y_train == 1]

ax.scatter(norm_data_norm[:, 0], norm_data_norm[:, 1], label='NORM', alpha=0.6, s=100)
ax.scatter(mi_data_norm[:, 0], mi_data_norm[:, 1], label='MI', alpha=0.6, s=100)
ax.set_xlabel(feature_names[0] + ' (normalized)')
ax.set_ylabel(feature_names[1] + ' (normalized)')
ax.set_title('Feature Distribution (After Z-Score Normalization) - 500 Hz')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_train, y_pred_train)
im = ax.imshow(cm, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Training Confusion Matrix - 500 Hz')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['NORM', 'MI'])
ax.set_yticklabels(['NORM', 'MI'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w", fontsize=16)

plt.colorbar(im, ax=ax)

# Plot 4: Feature importance
ax = axes[1, 1]
ax.barh(feature_names, result.importances_mean)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Permutation) - 500 Hz')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f'training_analysis{OUTPUT_SUFFIX}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nModel files saved to: {OUTPUT_DIR}/")
print(f"  - svm_model{OUTPUT_SUFFIX}.pkl")
print(f"  - scaler{OUTPUT_SUFFIX}.pkl")
print(f"  - model_metadata{OUTPUT_SUFFIX}.json")
print(f"  - training_data{OUTPUT_SUFFIX}.json")
print(f"  - training_analysis{OUTPUT_SUFFIX}.png")
