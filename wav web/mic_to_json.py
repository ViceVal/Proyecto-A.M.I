import sounddevice as sd
import numpy as np
import json
import scipy.io.wavfile as wav
#si quieres verlo pon en el archivo waveform.html abrir con live server (nesecitas descargar la estension vs code)

DURACION = 5  # segundos a grabar
FS = 44100   # frecuencia de muestreo

print("🎤 Grabando audio...")

audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()

print("✔ Grabación terminada")

# Guardar WAV
wav.write("grabacion.wav", FS, audio)

# Convertir a JSON
samples = (audio[:,0] * 32767).astype(np.int16).tolist()

with open("audio_data.json", "w") as f:
    json.dump(samples, f)

print("✔ Archivo JSON generado: audio_data.json")

print("✔ Archivo WAV generado: grabacion.wav")

