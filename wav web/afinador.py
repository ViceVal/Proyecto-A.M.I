import sounddevice as sd
import numpy as np
import math
import time
import os
#la version de web se abre con live server el tuner.html y la en codigo solo seria iniciarlo
# Frecuencias estándar de guitarra E A D G B E
notes = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63
}

FS = 44100
BLOCK = 2048

# -------------------------------
#     FUNCIONES DE PROCESO
# -------------------------------

def get_frequency(block, fs):
    window = block * np.hamming(len(block))
    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1/fs)
    magnitude = np.abs(fft)
    peak = np.argmax(magnitude)
    return freqs[peak]


def find_closest_note(freq):
    closest = None
    min_diff = 999999
    for note, f in notes.items():
        diff = abs(f - freq)
        if diff < min_diff:
            closest = note
            min_diff = diff
    return closest, notes[closest]


def cents_error(freq, target):
    return 1200 * math.log2(freq / target)


def draw_meter(cents):
    max_c = 50
    c = max(min(cents, max_c), -max_c)
    pos = int((c + max_c) / (2 * max_c) * 40)

    # Zona afinada ±5 cents
    perfect_left = int((max_c - 5) / (2 * max_c) * 40)
    perfect_right = int((max_c + 5) / (2 * max_c) * 40)

    meter = "|"

    for i in range(41):
        if i == pos:
            meter += "▲"  # aguja
        elif perfect_left <= i <= perfect_right:
            meter += "="  # zona afinada
        else:
            meter += "-"

    meter += "|"

    print(meter)
    print(" GRAVE <---------------------> AGUDO")


# -------------------------------
#            MENÚ
# -------------------------------

print("=== AFINADOR DE GUITARRA ===")
print("1) Iniciar afinador")
print("Presiona 1 y ENTER para comenzar...")
print("Para detener el afinador usa:   Ctrl + C")
print()

op = input("Opción: ")

if op != "1":
    print("Saliendo...")
    exit()

print("\nIniciando afinador en 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

print("\n🎤 Afinador activo, toca una cuerda. (Ctrl + C para detener)")

# -------------------------------
#         LOOP PRINCIPAL
# -------------------------------

with sd.InputStream(channels=1, samplerate=FS, blocksize=BLOCK):
    while True:
        audio = sd.rec(BLOCK, samplerate=FS, channels=1, dtype='float32')
        sd.wait()

        block = audio[:, 0]
        freq = get_frequency(block, FS)

        if freq <= 1:
            continue

        note, target_freq = find_closest_note(freq)
        cents = cents_error(freq, target_freq)

        os.system('cls' if os.name == 'nt' else 'clear')

        print("🎸 AFINADOR EN TIEMPO REAL (Ctrl + C para detener)\n")
        print(f"Frecuencia detectada: {freq:.2f} Hz")
        print(f"Cuerda más cercana: {note}  ({target_freq} Hz)")

        if cents > 5:
            state = "AGUDO → afloja"
        elif cents < -5:
            state = "GRAVE → aprieta"
        else:
            state = "✔ AFINADO"

        print(f"Desviación: {cents:+.1f} cents  |  {state}")

        print("\nIndicador:")
        draw_meter(cents)


        time.sleep(0.5)
