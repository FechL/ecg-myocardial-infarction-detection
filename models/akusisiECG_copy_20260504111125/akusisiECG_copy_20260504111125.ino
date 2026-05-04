// Konfigurasi Pin dan Sampling
#define ECG_INPUT A0
#define ECG_LO_PLUS 11
#define ECG_LO_MINUS 10
#define SAMPLE_INTERVAL 10 // 100 Hz = 10 ms interval

void setup() {
    Serial.begin(115200); 
    pinMode(ECG_LO_PLUS, INPUT);
    pinMode(ECG_LO_MINUS, INPUT);
    pinMode(ECG_INPUT, INPUT);
}
  
void loop() {
    static unsigned long lastSampleTime = 0;
    unsigned long currentTime = millis();

    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;

        // Cek Lead-Off
        if (digitalRead(ECG_LO_PLUS) == HIGH || digitalRead(ECG_LO_MINUS) == HIGH) {
            // Kirim 0.0 agar Python tidak error saat plotting
            Serial.println(0.0, 4); 
        } 
        else {
            int raw_adc = analogRead(ECG_INPUT);
            
            // Perbaikan rumus: 
            // (raw_adc / 1023.0 * 5.0) membaca voltase sebenarnya di pin A0
            // Dikurangi 1.65 (VCC/2) untuk menaruh baseline di titik 0
            float voltage = (raw_adc / 1023.0) * 5.0 - 1.65;

            Serial.println(voltage, 4); 
        }
    }
}