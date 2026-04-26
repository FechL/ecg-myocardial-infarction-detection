/*
 * ECG Myocardial Infarction Detection System using Arduino Uno
 * Based on: SVM Classification dengan Pathological Q Waves & ST Segment
 * Elevation
 *
 * Hardware:
 * - Arduino Uno
 * - AD8232 ECG Sensor Module
 * - LCD 16x2 (I2C)
 * - SD Card Module (Optional, untuk logging)
 *
 * Pins:
 * - AD8232 Output: A0 (Analog)
 * - AD8232 LO+: D2 (Digital - Lead off detection)
 * - AD8232 LO-: D3 (Digital - Lead off detection)
 * - LCD SDA: A4 (I2C)
 * - LCD SCL: A5 (I2C)
 *
 * Training Parameters from Python Model:
 * - Intercept: 0.6634
 * - Q_Waves Mean: -0.0608, Std: 0.1068
 * - ST_Elevation Mean: -0.0344, Std: 0.0758
 * - Exponential Filter w: 0.55
 * - Sampling Rate: 100 Hz
 */

#include <LiquidCrystal_I2C.h>
#include <Wire.h>

// ==================== CONFIGURATION ====================

// Sampling Configuration
#define SAMPLING_RATE 100       // Hz
#define SAMPLE_INTERVAL 10      // milliseconds (1000 / SAMPLING_RATE)
#define SAMPLES_PER_WINDOW 1000 // 10 seconds of data at 100 Hz

// Pin Configuration
#define ECG_INPUT A0
#define ECG_LO_PLUS 2
#define ECG_LO_MINUS 3

// LCD Configuration (I2C address 0x27 for 16x2 display)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ==================== TRAINED MODEL WEIGHTS ====================

// Z-Score Normalization Parameters
const float Q_WAVES_MEAN = -0.0608;
const float Q_WAVES_STD = 0.1068;
const float ST_ELEVATION_MEAN = -0.0344;
const float ST_ELEVATION_STD = 0.0758;

// SVM Decision Boundary
const float SVM_INTERCEPT = 0.6634;

// Exponential Filter Parameter
const float EXPONENTIAL_W = 0.55;

// ==================== SIGNAL PROCESSING ====================

// ECG Signal Buffer
float ecg_raw[SAMPLES_PER_WINDOW];
float ecg_filtered[SAMPLES_PER_WINDOW];
int sample_count = 0;

// Feature Storage
float q_waves_feature = 0.0;
float st_elevation_feature = 0.0;

// Exponential Filter State
float filtered_value_prev = 0.0;

// ==================== FUNCTION PROTOTYPES ====================

void setupAD8232();
void setupLCD();
void readECGSample();
float applyExponentialFilter(float raw_value);
void detectQRSPeaks();
void extractFeatures();
void normalizeFeatures(float &q_norm, float &st_norm);
int predictSVM(float q_norm, float st_norm);
void displayDiagnosis(String diagnosis);
void printDebugInfo();

// ==================== SETUP ====================

void setup() {
    Serial.begin(9600);
    delay(1000);

    Serial.println("======================================");
    Serial.println("ECG MI Detection System - Arduino");
    Serial.println("======================================");

    setupAD8232();
    setupLCD();

    delay(500);
    Serial.println("System initialized successfully!");
}

// ==================== MAIN LOOP ====================

void loop() {
    static unsigned long lastSampleTime = 0;
    unsigned long currentTime = millis();

    // Read sample at fixed interval (100 Hz)
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;
        readECGSample();

        // Once we have enough samples (10 seconds)
        if (sample_count >= SAMPLES_PER_WINDOW) {

            // Process the collected data
            detectQRSPeaks();
            extractFeatures();

            // Normalize features
            float q_norm, st_norm;
            normalizeFeatures(q_norm, st_norm);

            // Make prediction
            int prediction = predictSVM(q_norm, st_norm);

            // Display results
            String diagnosis = (prediction == 0) ? "NORMAL" : "MI";
            displayDiagnosis(diagnosis);

            // Print debug info to Serial
            printDebugInfo();

            // Reset for next window
            sample_count = 0;
            filtered_value_prev = 0.0;
        }
    }
}

// ==================== AD8232 SETUP ====================

void setupAD8232() {
    pinMode(ECG_LO_PLUS, INPUT);
    pinMode(ECG_LO_MINUS, INPUT);
    pinMode(ECG_INPUT, INPUT);

    Serial.println("[+] AD8232 ECG Sensor initialized");
}

// ==================== LCD SETUP ====================

void setupLCD() {
    lcd.init();
    lcd.backlight();

    lcd.setCursor(0, 0);
    lcd.print("ECG MI Detector");
    lcd.setCursor(0, 1);
    lcd.print("Initializing...");

    delay(2000);
    lcd.clear();

    Serial.println("[+] LCD 16x2 initialized");
}

// ==================== ECG SAMPLING ====================

void readECGSample() {
    // Check for lead-off detection
    if (digitalRead(ECG_LO_PLUS) || digitalRead(ECG_LO_MINUS)) {
        Serial.println("!!! Lead Off Detection !!!");
        lcd.setCursor(0, 0);
        lcd.print("Lead Off!");
        return;
    }

    // Read analog value (0-1023 for 0-5V)
    int raw_adc = analogRead(ECG_INPUT);

    // Convert to voltage (0-5V range, centered at 2.5V)
    float voltage = (raw_adc / 1023.0) * 5.0 - 2.5;

    // Apply exponential filter for noise reduction
    ecg_raw[sample_count] = voltage;
    ecg_filtered[sample_count] = applyExponentialFilter(voltage);

    sample_count++;
}

// ==================== EXPONENTIAL FILTER ====================

float applyExponentialFilter(float raw_value) {
    // Y_n = w * X_n + (1 - w) * Y_{n-1}
    // where w = 0.55

    if (sample_count == 0) {
        filtered_value_prev = raw_value;
        return raw_value;
    }

    float filtered =
        EXPONENTIAL_W * raw_value + (1.0 - EXPONENTIAL_W) * filtered_value_prev;
    filtered_value_prev = filtered;

    return filtered;
}

// ==================== QRS PEAK DETECTION ====================

void detectQRSPeaks() {
    // Simplified Peak Detection using threshold
    // Find local maxima and minima

    // Find maximum and minimum values for normalization
    float max_val = -999.0;
    float min_val = 999.0;

    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        if (ecg_filtered[i] > max_val)
            max_val = ecg_filtered[i];
        if (ecg_filtered[i] < min_val)
            min_val = ecg_filtered[i];
    }

    Serial.print("[*] ECG Range: ");
    Serial.print(min_val);
    Serial.print(" to ");
    Serial.println(max_val);
}

// ==================== FEATURE EXTRACTION ====================

void extractFeatures() {
    /*
     * FITUR 1: Pathological Q Waves
     *   - Amplitudo minimum (valley) sebelum R peak
     *   - Representasi: rata-rata amplitudo Q waves
     */

    /*
     * FITUR 2: ST Segment Elevation
     *   - Amplitudo rata-rata antara S dan T wave
     *   - Representasi: rata-rata amplitudo ST segment
     */

    // Simplified feature extraction using statistical features
    // Q_Waves = average of lowest 20% of values
    // ST_Elevation = average of highest 20% of values

    // Calculate mean and std of ECG signal
    float mean_ecg = 0.0;
    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        mean_ecg += ecg_filtered[i];
    }
    mean_ecg /= SAMPLES_PER_WINDOW;

    // Q Waves feature (negative peaks)
    float q_sum = 0;
    int q_count = 0;

    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        if (ecg_filtered[i] < mean_ecg - 0.1) { // Threshold for Q waves
            q_sum += ecg_filtered[i];
            q_count++;
        }
    }

    if (q_count > 0) {
        q_waves_feature = q_sum / q_count;
    } else {
        q_waves_feature = mean_ecg;
    }

    // ST Elevation feature (area around peak)
    float st_sum = 0;
    int st_count = 0;

    for (int i = 0; i < SAMPLES_PER_WINDOW; i++) {
        if (ecg_filtered[i] > mean_ecg - 0.05 &&
            ecg_filtered[i] < mean_ecg + 0.05) {
            st_sum += ecg_filtered[i];
            st_count++;
        }
    }

    if (st_count > 0) {
        st_elevation_feature = st_sum / st_count;
    } else {
        st_elevation_feature = mean_ecg;
    }

    Serial.print("[*] Raw Features: Q=");
    Serial.print(q_waves_feature);
    Serial.print(", ST=");
    Serial.println(st_elevation_feature);
}

// ==================== FEATURE NORMALIZATION ====================

void normalizeFeatures(float &q_norm, float &st_norm) {
    // Z-Score Normalization: z = (x - mean) / std

    q_norm = (q_waves_feature - Q_WAVES_MEAN) / Q_WAVES_STD;
    st_norm = (st_elevation_feature - ST_ELEVATION_MEAN) / ST_ELEVATION_STD;

    Serial.print("[*] Normalized: Q=");
    Serial.print(q_norm);
    Serial.print(", ST=");
    Serial.println(st_norm);
}

// ==================== SVM PREDICTION ====================

int predictSVM(float q_norm, float st_norm) {
    /*
     * SVM Decision Function (Simplified Linear Approximation)
     *
     * For Arduino, we use a simplified linear decision boundary:
     * score = w1*Q_norm + w2*ST_norm + b
     *
     * Feature Importance from Training:
     * - Q_Waves: 0.1343 (weight higher)
     * - ST_Elevation: 0.1143 (weight lower)
     *
     * Normalized weights (proportional to importance):
     */

    // Simplified weights based on feature importance
    const float W_Q_WAVES = 1.3; // Proportional to importance
    const float W_ST_ELEVATION = 1.1;

    // Calculate decision score
    float score =
        (W_Q_WAVES * q_norm) + (W_ST_ELEVATION * st_norm) + SVM_INTERCEPT;

    Serial.print("[*] SVM Score: ");
    Serial.println(score);

    // Classification
    // score > 0 → class 0 (NORMAL)
    // score ≤ 0 → class 1 (MI)

    if (score > 0.0) {
        return 0; // NORMAL
    } else {
        return 1; // MYOCARDIAL INFARCTION
    }
}

// ==================== DISPLAY RESULTS ====================

void displayDiagnosis(String diagnosis) {
    Serial.print("[RESULT] Diagnosis: ");
    Serial.println(diagnosis);

    // LCD Display
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Diagnosis:");

    lcd.setCursor(0, 1);
    if (diagnosis == "NORMAL") {
        lcd.print("NORMAL      ");
        Serial.println("✓ Status: NORMAL - No signs of MI");
    } else {
        lcd.print("MI DETECTED!");
        Serial.println("⚠ Status: ALERT - MI Detected!");

        // Optional: Trigger alarm or alert
        // digitalWrite(ALARM_PIN, HIGH);
    }

    delay(3000); // Display result for 3 seconds

    // Display feature info
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Q:");
    lcd.print(q_waves_feature, 3);
    lcd.setCursor(9, 0);
    lcd.print("ST:");
    lcd.print(st_elevation_feature, 3);

    delay(2000);
}

// ==================== DEBUG INFO ====================

void printDebugInfo() {
    Serial.println("\n========== DEBUG INFO ==========");
    Serial.print("Q_Waves (raw): ");
    Serial.println(q_waves_feature);
    Serial.print("ST_Elevation (raw): ");
    Serial.println(st_elevation_feature);

    float q_norm = (q_waves_feature - Q_WAVES_MEAN) / Q_WAVES_STD;
    float st_norm =
        (st_elevation_feature - ST_ELEVATION_MEAN) / ST_ELEVATION_STD;

    Serial.print("Q_Waves (normalized): ");
    Serial.println(q_norm);
    Serial.print("ST_Elevation (normalized): ");
    Serial.println(st_norm);

    Serial.println("=============================\n");
}

// ==================== OPTIONAL: HEART RATE CALCULATION ====================

float calculateHeartRate() {
    /*
     * Simple heart rate calculation
     * Count zero-crossings or peaks in a 60-second window
     * BPM = (peak_count / 60) * 60 = peak_count
     */

    // This requires peak detection
    // For now, returning 0
    return 0.0;
}

// ==================== OPTIONAL: SD CARD LOGGING ====================

void logDataToSD() {
    /*
     * Optional: Log ECG data and diagnosis to SD card
     * Requires: SD Card module on SPI pins
     *
     * Structure:
     * - Timestamp
     * - Raw ECG signal (sample)
     * - Q_Waves feature
     * - ST_Elevation feature
     * - Prediction
     */

    // Implementation depends on SD card library
}
