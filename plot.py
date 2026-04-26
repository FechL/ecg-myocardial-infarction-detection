import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Path ke folder dataset
DATASET_PATH = 'dataset'
DB_PATH = os.path.join(DATASET_PATH, 'ptbxl_database.csv')
SCP_PATH = os.path.join(DATASET_PATH, 'scp_statements.csv')

def load_database():
    """
    Membaca metadata database dari ptbxl_database.csv
    
    Returns:
        DataFrame dengan metadata atau None jika file tidak ada
    """
    if os.path.exists(DB_PATH):
        try:
            df = pd.read_csv(DB_PATH)
            print(f"✓ Database loaded: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading database: {e}")
            return None
    else:
        print(f"Database file not found at {DB_PATH}")
        return None

def load_scp_statements():
    """
    Membaca SCP diagnostic statements
    
    Returns:
        DataFrame dengan SCP statements atau None
    """
    if os.path.exists(SCP_PATH):
        try:
            df = pd.read_csv(SCP_PATH)
            print(f"✓ SCP statements loaded: {len(df)} statements")
            return df
        except Exception as e:
            print(f"Error loading SCP statements: {e}")
            return None
    else:
        print(f"SCP statements file not found at {SCP_PATH}")
        return None

def get_record_info_from_db(ecg_id, db_df):
    """
    Mendapatkan informasi record dari database
    
    Args:
        ecg_id: ID ECG
        db_df: Database DataFrame
    
    Returns:
        Row data atau None
    """
    if db_df is not None:
        mask = db_df['ecg_id'] == ecg_id
        if mask.any():
            return db_df[mask].iloc[0]
    return None

def load_ecg_record(record_name):
    """
    Membaca file ECG dari dataset
    
    Args:
        record_name: Nama file tanpa extension (misal: '05469_lr')
    
    Returns:
        record: Objek record dari wfdb
    """
    try:
        record_path = os.path.join(DATASET_PATH, record_name)
        record = wfdb.rdrecord(record_path)
        return record
    except Exception as e:
        print(f"Error membaca record {record_name}: {e}")
        return None

def plot_single_ecg(record_name, db_df=None, figsize=(14, 8)):
    """
    Menampilkan plot ECG dari satu record dengan informasi metadata
    
    Args:
        record_name: Nama file tanpa extension
        db_df: Database DataFrame untuk metadata (opsional)
        figsize: Ukuran figure (width, height)
    """
    record = load_ecg_record(record_name)
    if record is None:
        return
    
    # Extract ECG ID dari record name (misal '05469_lr' -> 5469)
    try:
        ecg_id = int(record_name.split('_')[0])
    except:
        ecg_id = None
    
    print(f"\n{'='*50}")
    print(f"Record: {record_name}")
    print(f"{'='*50}")
    print(f"Sampling rate: {record.fs} Hz")
    print(f"Number of leads: {len(record.sig_name)}")
    print(f"Lead names: {record.sig_name}")
    print(f"Duration: {record.sig_len / record.fs:.2f} seconds")
    print(f"Total samples: {record.sig_len}")
    
    # Tampilkan info dari database jika tersedia
    if db_df is not None and ecg_id is not None:
        record_info = get_record_info_from_db(ecg_id, db_df)
        if record_info is not None:
            print(f"\n--- Metadata ---")
            print(f"Patient ID: {record_info.get('patient_id', 'N/A')}")
            print(f"Age: {record_info.get('age', 'N/A')}")
            print(f"Sex: {record_info.get('sex', 'N/A')}")
            print(f"Diagnosis/SCP codes: {record_info.get('scp_codes', 'N/A')}")
    
    # Plot
    fig, axes = plt.subplots(len(record.sig_name), 1, figsize=figsize, sharex=True)
    
    # Handle jika hanya 1 signal
    if len(record.sig_name) == 1:
        axes = [axes]
    
    # Buat time array dalam detik
    time = np.arange(record.sig_len) / record.fs
    
    for i, signal_name in enumerate(record.sig_name):
        axes[i].plot(time, record.p_signal[:, i], linewidth=0.7, color='steelblue')
        axes[i].set_ylabel(signal_name, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)', fontweight='bold')
    
    title = f'ECG Record: {record_name}'
    if ecg_id and db_df is not None:
        record_info = get_record_info_from_db(ecg_id, db_df)
        if record_info is not None:
            title += f' | Age: {record_info.get("age", "?")} | Sex: {record_info.get("sex", "?")}'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_all_ecg_records(figsize=(14, 6), db_df=None):
    """
    Menampilkan plot semua ECG records dalam dataset
    
    Args:
        figsize: Ukuran figure
        db_df: Database DataFrame (opsional)
    """
    # Cari semua file .hea di direktori dataset (recursive)
    hea_files = sorted(Path(DATASET_PATH).rglob('*.hea'))
    
    if not hea_files:
        print("Tidak ada file .hea ditemukan di dataset folder")
        return
    
    print(f"\nDitemukan {len(hea_files)} record ECG")
    
    for hea_file in hea_files:
        record_name = hea_file.stem  # Nama file tanpa extension
        plot_single_ecg(record_name, db_df, figsize)

def plot_multiple_ecg(record_names, db_df=None, figsize=(14, 10)):
    """
    Menampilkan plot beberapa ECG records dalam satu figure
    
    Args:
        record_names: List nama file tanpa extension
        db_df: Database DataFrame (opsional)
        figsize: Ukuran figure
    """
    fig, axes = plt.subplots(len(record_names), 1, figsize=figsize, sharex=False)
    
    if len(record_names) == 1:
        axes = [axes]
    
    for idx, record_name in enumerate(record_names):
        record = load_ecg_record(record_name)
        if record is None:
            continue
        
        time = np.arange(record.sig_len) / record.fs
        
        # Plot channel pertama (biasanya yang paling informatif)
        axes[idx].plot(time, record.p_signal[:, 0], linewidth=0.7, color='steelblue')
        axes[idx].set_ylabel(record.sig_name[0] if record.sig_name else 'ECG', fontweight='bold')
        axes[idx].set_title(f'{record_name} - Fs: {record.fs} Hz')
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)', fontweight='bold')
    plt.suptitle('Comparison of Multiple ECG Records', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("="*60)
    print("PTB-XL ECG Dataset Visualization")
    print("="*60)
    
    # Load database jika ada
    db_df = load_database()
    scp_df = load_scp_statements()
    
    print("\n" + "="*60)
    print("OPTION 1: Plot satu record dengan metadata")
    print("="*60)
    plot_single_ecg('05469_lr', db_df=db_df)
    
    print("\n" + "="*60)
    print("OPTION 2: Plot beberapa records untuk perbandingan")
    print("="*60)
    plot_multiple_ecg(['05469_lr', '09514_lr'], db_df=db_df)
    
    print("\n" + "="*60)
    print("OPTION 3: Plot semua records di dataset")
    print("="*60)
    # Uncomment jika ingin plot semua:
    # plot_all_ecg_records(db_df=db_df)
