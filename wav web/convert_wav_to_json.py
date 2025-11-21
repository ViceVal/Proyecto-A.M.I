import wave
import json
import audioop
#si quieres verlo pon en el archivo waveform.html abrir con live server (nesecitas descargar la estension de vs code)
wav_file = "audio.wav" #pon el nombre de tu audio hay
json_file = "audio_data.json"

with wave.open(wav_file, 'rb') as wav:
    n_frames = wav.getnframes()
    sample_width = wav.getsampwidth()     # tamaño original
    channels = wav.getnchannels()

    frames = wav.readframes(n_frames)

    # Convertir a 16 bits PCM si NO lo está
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2

    samples = []

    # Si el wav es estéreo → usar canal izquierdo
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 1, 0)

    # Convertir bytes a enteros
    for i in range(0, len(frames), 2):
        frame = frames[i:i + 2]
        if len(frame) < 2:
            continue
        value = int.from_bytes(frame, byteorder='little', signed=True)
        samples.append(value)

# Guardar JSON
with open(json_file, "w") as f:
    json.dump(samples, f)

print("✔ Archivo convertido correctamente a audio_data.json")

