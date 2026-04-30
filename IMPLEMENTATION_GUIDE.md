# ECG MI Detector - Implementation Versions

## Ringkasan

Project ini memiliki 3 versi implementasi sesuai dengan hardware dan memory constraints:

### 1. **Arduino Uno** (ECG_MI_Detector.ino)
- **Processor**: ATmega328P (8-bit)
- **RAM**: ~2 KB available (after bootloader and libraries)
- **Sampling Rate**: 100 Hz
- **Samples per Window**: 200 (2 seconds)
- **Prediction Frequency**: Every 2 seconds
- **Use Case**: Resource-constrained deployment
- **Output**: Serial Monitor + Serial Plotter

**Spesifikasi Teknis:**
```
Buffer Size = 200 samples × 4 bytes/float × 2 buffers = 1.6 KB ✓
Total Memory Usage ≈ 2 KB (within Arduino limits)
```

**Keuntungan:**
- Sangat hemat power
- Cost-effective
- Bisa deploy di berbagai Arduino boards

**Keterbatasan:**
- Window yang lebih pendek (2 detik) dapat mengurangi akurasi
- Processing terbatas
- Tidak ada WiFi/Bluetooth

---

### 2. **Arduino + Train 500 Hz Model** (train_500hz.py)
- **Sampling Rate**: 500 Hz
- **Samples per Window**: 200 (0.4 seconds)
- **Use Case**: Untuk testing berbagai window sizes
- **File Output**: 
  - `svm_model_500hz.pkl`
  - `scaler_500hz.pkl`
  - `model_metadata_500hz.json`

**Catatan**: Model ini dapat digunakan untuk Arduino jika ingin mengurangi computational overhead dengan window yang lebih pendek, tapi dengan sampling rate yang lebih tinggi.

---

### 3. **ESP32** (ECG_MI_Detector_ESP32.ino)
- **Processor**: Xtensa 32-bit dual-core (240 MHz)
- **RAM**: 520 KB SRAM
- **Sampling Rate**: 100 Hz (dapat dinaikkan ke 500 Hz)
- **Samples per Window**: 1000 (10 seconds)
- **Prediction Frequency**: Every 10 seconds
- **Use Case**: Production deployment dengan akurasi lebih tinggi
- **Output**: Serial Monitor + Serial Plotter
- **Bonus Features**: WiFi, Bluetooth, SD Card support (dapat ditambahkan)

**Spesifikasi Teknis:**
```
Buffer Size = 1000 samples × 4 bytes/float × 2 buffers = 8 KB ✓
Total Memory Usage ≈ 50-100 KB (masih di bawah 520 KB limit)
```

**Keuntungan:**
- Akurasi lebih tinggi dengan window lebih panjang
- Banyak memory untuk fitur tambahan
- WiFi connectivity untuk telemetry
- Better ADC resolution (12-bit vs 10-bit)
- Faster processing

---

## Perbandingan Hardware

| Parameter | Arduino Uno | ESP32 |
|-----------|------------|-------|
| Processor | ATmega328P | Xtensa 32-bit |
| Clock Speed | 16 MHz | 240 MHz |
| RAM Total | 2 KB | 520 KB |
| RAM Available | ~1.5 KB | ~400 KB |
| ADC Resolution | 10-bit | 12-bit |
| WiFi | ✗ | ✓ |
| Bluetooth | ✗ | ✓ |
| Cost | ~$3 | ~$5-10 |
| Power Usage | ~50 mA | ~80-160 mA |

---

## Serial Output Format

Kedua versi menggunakan format yang sama untuk Serial Plotter:

```
Raw_Signal,Filtered_Signal
0.1234,0.1156
0.2345,0.2087
0.3456,0.3018
...
```

**Menggunakan Serial Plotter di Arduino IDE:**
1. Upload kode ke board
2. Buka `Tools → Serial Plotter`
3. Set baud rate sesuai (#define SERIAL_BAUD)
4. Lihat real-time plot dari Raw dan Filtered signals

---

## Model Training

### Versi 100 Hz (Original)
```bash
python train.py
```
- Output: `svm_model.pkl`, `scaler.pkl`, `model_metadata.json`
- Samples per window: 1000 (10 seconds)
- Gunakan di: Arduino (dengan adjustment) atau ESP32

### Versi 500 Hz (Baru)
```bash
python train_500hz.py
```
- Output: `svm_model_500hz.pkl`, `scaler_500hz.pkl`, `model_metadata_500hz.json`
- Samples per window: 200 (0.4 seconds)
- Gunakan di: Arduino (jika ingin shorter window)

---

## Implementasi di Hardware

### Step 1: Pilih Hardware
- **Untuk prototype/testing**: Arduino Uno
- **Untuk production**: ESP32

### Step 2: Update Model Weights (jika diperlukan)
Edit file `.ino` dan update constants:
```cpp
const float Q_WAVES_MEAN = -0.0608;        // dari model_metadata.json
const float Q_WAVES_STD = 0.1068;
const float ST_ELEVATION_MEAN = -0.0344;
const float ST_ELEVATION_STD = 0.0758;
const float SVM_INTERCEPT = 0.6634;
```

### Step 3: Upload ke Board
```
1. Arduino IDE → Sketch → Upload
2. Atau: Tools → Serial Plotter (untuk visualisasi)
```

### Step 4: Monitor
```
1. Tools → Serial Monitor
2. Set baud rate: 115200
3. Lihat debug output dan diagnosis
```

---

## Memory Analysis

### Arduino Uno (200 samples)
```
Buffers:
  - ecg_raw[200]:      200 × 4 = 800 bytes
  - ecg_filtered[200]: 200 × 4 = 800 bytes
  
Global Variables:      ~200 bytes
Stack/Heap:            ~400 bytes

Total:                 ~2.2 KB (AVAILABLE!)
```

### ESP32 (1000 samples)
```
Buffers:
  - ecg_raw[1000]:      1000 × 4 = 4 KB
  - ecg_filtered[1000]: 1000 × 4 = 4 KB
  
Global Variables:       ~200 bytes
Stack/Heap:             ~40 KB

Total:                  ~48 KB (out of 520 KB available - 9% usage)
```

---

## Troubleshooting

### Arduino
- **Error: "Sketch uses 30526 bytes... only 30720 available"**
  - Solution: Reduce SERIAL_BAUD atau disable debugging
  
- **Prediction unstable**
  - Reason: Short window (2 seconds) dapat terpengaruh noise
  - Solution: Gunakan ESP32 dengan window lebih panjang

### ESP32
- **Error: "Brownout detector was triggered"**
  - Solution: Gunakan power supply yang lebih baik (≥1A)
  
- **Serial data garbled**
  - Solution: Check SERIAL_BAUD setting di kode vs Serial Monitor

---

## Next Steps

1. **Train model** dengan kedua sampling rates:
   ```bash
   python train.py        # 100 Hz (original)
   python train_500hz.py  # 500 Hz (new)
   ```

2. **Extract weights** dari `model_metadata.json`:
   ```json
   {
     "scaler_mean": [-0.0608, -0.0344],
     "scaler_std": [0.1068, 0.0758],
     "svm_intercept": 0.6634
   }
   ```

3. **Update `.ino` files** dengan weights terbaru

4. **Test di hardware**:
   - Arduino: untuk quick testing
   - ESP32: untuk production deployment

---

## File Organization

```
├── train.py                      # Training script (100 Hz, 1000 samples)
├── train_500hz.py               # Training script (500 Hz, 200 samples)
├── ECG_MI_Detector.ino           # Arduino Uno version (200 samples)
├── ECG_MI_Detector_ESP32.ino     # ESP32 version (1000 samples)
├── models/
│   ├── svm_model.pkl            # Model (100 Hz)
│   ├── svm_model_500hz.pkl      # Model (500 Hz)
│   ├── scaler.pkl               # Scaler (100 Hz)
│   ├── scaler_500hz.pkl         # Scaler (500 Hz)
│   ├── model_metadata.json      # Metadata (100 Hz)
│   └── model_metadata_500hz.json# Metadata (500 Hz)
└── dataset/
    └── records100/              # ECG records
```

---

## Catatan Penting

1. **Memory adalah bottleneck utama di Arduino**
   - Tidak bisa menggunakan dynamic memory allocation
   - Harus hardcode buffer sizes

2. **Model weights perlu diupdate di `.ino` files**
   - Setiap kali train model baru
   - Extract dari `model_metadata.json`

3. **Serial Plotter sangat berguna untuk debugging**
   - Visualisasi real-time signal
   - Deteksi masalah lead-off
   - Monitor signal quality

4. **Window size mempengaruhi akurasi**
   - Lebih panjang = lebih akurat tapi lebih lambat
   - Arduino: 200 samples = 2 seconds (trade-off)
   - ESP32: 1000 samples = 10 seconds (lebih akurat)
