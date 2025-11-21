import os

print("=== SELECCIONE MODO DE AUDIO ===")
print("1) Archivo WAV")
print("2) Micrófono (tiempo real)")

op = input("Opción: ")

if op == "1":
    os.system("python convert_wav_to_json.py")
elif op == "2":
    os.system("python mic_to_json.py")
else:
    print("Opción no válida")