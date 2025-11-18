#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <driver/i2s.h>
#include <math.h>

// =========================
// PINOUT MATRIX
// =========================
int filas[]    = {32, 33, 25, 26};
int columnas[] = {27, 14, 12, 13};
int numFilas = sizeof(filas)/sizeof(filas[0]);
int numColumnas = sizeof(columnas)/sizeof(columnas[0]);
bool estadoPrevio[4][4] = {false};  // Debounce

// =========================
// I2S MAX98357A
// =========================
#define I2S_BCLK 18
#define I2S_LRC  19
#define I2S_DOUT 22

void i2s_init() {
  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 44100,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_LRC,
    .data_out_num = I2S_DOUT,
    .data_in_num = -1
  };

  i2s_driver_install(I2S_NUM_0, &config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

// =========================
// REPRODUCIR TONO
// =========================
void playTone(float freq, int duration_ms) {
  const int sampleRate = 44100;
  int totalSamples = (sampleRate * duration_ms) / 1000;
  int16_t sample;
  size_t written;

  for (int i = 0; i < totalSamples; i++) {
    float t = (float)i / sampleRate;
    sample = (int16_t)(sin(2 * PI * freq * t) * 30000);
    i2s_write(I2S_NUM_0, &sample, sizeof(sample), &written, portMAX_DELAY);
  }
}

// =========================
// LCD 16x2 I2C
// =========================
LiquidCrystal_I2C lcd(0x27, 16, 2);  // Cambia 0x27 si tu módulo tiene otra dirección

// =========================
// SETUP
// =========================
void setup() {
  Serial.begin(115200);
  Serial.println("Matriz + I2S + LCD listo");

  // Inicializar LCD
  Wire.begin(21, 23); // SDA, SCL
  delay(100);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Presiona un boton");

  // Inicializar matriz
  for (int i = 0; i < numFilas; i++) {
    pinMode(filas[i], OUTPUT);
    digitalWrite(filas[i], HIGH);
  }
  for (int j = 0; j < numColumnas; j++) {
    pinMode(columnas[j], INPUT_PULLUP);
  }

  i2s_init();
}

// =========================
// LOOP
// =========================
void loop() {
  for (int f = 0; f < numFilas; f++) {
    digitalWrite(filas[f], LOW);
    for (int c = 0; c < numColumnas; c++) {
      bool presionado = (digitalRead(columnas[c]) == LOW);

      if (presionado && !estadoPrevio[f][c]) {
        Serial.print("Boton F");
        Serial.print(f);
        Serial.print(" C");
        Serial.print(c);
        Serial.println(" presionado");

        // Asignar frecuencia y nota
        float freq = 0;
        String nota = "";

        if (f == 0 && c == 0) { freq = 440; nota = "A4"; }
        if (f == 0 && c == 1) { freq = 493.88; nota = "B4"; }
        if (f == 0 && c == 2) { freq = 523.25; nota = "C5"; }
        if (f == 0 && c == 3) { freq = 587.33; nota = "D5"; }

        if (f == 1 && c == 0) { freq = 659.25; nota = "E5"; }
        if (f == 1 && c == 1) { freq = 698.46; nota = "F5"; }
        if (f == 1 && c == 2) { freq = 783.99; nota = "G5"; }
        if (f == 1 && c == 3) { freq = 880;    nota = "A5"; }

        if (f == 2 && c == 0) { freq = 987.77; nota = "B5"; }
        if (f == 2 && c == 1) { freq = 1046.5; nota = "C6"; }
        if (f == 2 && c == 2) { freq = 1174.66; nota = "D6"; }
        if (f == 2 && c == 3) { freq = 1318.51; nota = "E6"; }

        if (f == 3 && c == 0) { freq = 1396.91; nota = "F6"; }
        if (f == 3 && c == 1) { freq = 1567.98; nota = "G6"; }
        if (f == 3 && c == 2) { freq = 1760;    nota = "A6"; }
        if (f == 3 && c == 3) { freq = 1975.53; nota = "B6"; }

        // Mostrar en LCD
        lcd.clear();
        lcd.setCursor(0,0);
        lcd.print("Nota:");
        lcd.setCursor(6,0);
        lcd.print(nota);

        if (freq > 0) playTone(freq, 1000); // Reproducir tono 2 segundos
      }

      estadoPrevio[f][c] = presionado;
    }
    digitalWrite(filas[f], HIGH);
  }
  delay(10);
}
