import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  # Clave para manejar carpetas
import sys               # Para salir si hay un error

# --- Funciones de conversión de notas (Sin cambios) ---

def hz_to_midi(f):
    """Convierte frecuencia en Hz a número de nota MIDI."""
    return 69.0 + 12.0 * np.log2(f / 440.0)

def midi_to_note_name(m):
    """Convierte número de nota MIDI a nombre (ej. A#4)."""
    m_rounded = int(round(m))
    pc = m_rounded % 12
    octave = (m_rounded // 12) - 1
    NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{NOTE_NAMES_SHARP[pc]}{octave}"

def hz_to_note_name(f):
    """Convierte frecuencia en Hz a nombre de nota."""
    if f is None or f <= 0 or not np.isfinite(f):
        return "N/A"
    m = hz_to_midi(f)
    return midi_to_note_name(m)

# --- Funciones de carga y espectrograma (Sin cambios) ---

def load_audio(path, sr=None, trim=True):
    """Carga un archivo de audio."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if trim:
        yt, _ = librosa.effects.trim(y, top_db=30)
        if len(yt) > 0:
            y = yt
    return y, sr

def compute_spectrogram(y, sr, n_fft=4096, hop_length=512):
    """Calcula el espectrograma en dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

# --- Función Principal de Análisis (Sin cambios en su lógica interna) ---

def analyze_f0_and_harmonics(audio_file_path, out_image_path):
    """
    Analiza UN archivo de audio y guarda UN gráfico de espectrograma.
    """
    print(f"Cargando audio: {audio_file_path.name}...")
    try:
        y, sr = load_audio(audio_file_path, sr=22050, trim=True)
        if len(y) == 0:
            print("  > Error: El audio está vacío o es silencio.")
            return
    except Exception as e:
        print(f"  > Error cargando el archivo de audio: {e}")
        return

    n_fft = 4096
    hop_length = 512
    
    # Estimar F0
    f0_series = librosa.yin(y,
                            fmin=librosa.note_to_hz('E2'),
                            fmax=librosa.note_to_hz('E6'),
                            sr=sr)
    f0_valid = f0_series[np.isfinite(f0_series) & (f0_series > 0)]
    
    if len(f0_valid) == 0:
        f0_estimate = None
        note_name = "N/A"
    else:
        f0_estimate = float(np.median(f0_valid))
        note_name = hz_to_note_name(f0_estimate)
        print(f"  > f0 estimada: {f0_estimate:.2f} Hz ({note_name})")

    # Calcular Espectrograma
    S_db = compute_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length)

    # Graficar
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    
    if f0_estimate:
        plt.title(f"Análisis de Armónicos: {audio_file_path.name}\n f0 ≈ {f0_estimate:.2f} Hz ({note_name})")
        max_harmonic = 8
        for n in range(1, max_harmonic + 1):
            harmonic_freq = f0_estimate * n
            if harmonic_freq > sr / 2: break
            plt.axhline(harmonic_freq, linestyle="--", color="red", linewidth=0.7, alpha=0.8)
            plt.text(S_db.shape[1] * hop_length / sr, harmonic_freq, 
                     f" H{n} ({harmonic_freq:.0f} Hz)", color="red",
                     ha="right", va="center", fontsize=8, backgroundcolor=(1,1,1,0.5))
    else:
        plt.title(f"Espectrograma: {audio_file_path.name} (f0 no detectada)")

    plt.ylim(0, 6000)
    plt.tight_layout()
    
    # La función analyze_f0... no necesita crear la carpeta,
    # porque la función main() ya lo hizo.
    plt.savefig(out_image_path, dpi=150)
    print(f"  > Gráfico guardado en: {out_image_path}")
    plt.close()

# --- *** NUEVA FUNCIÓN MAIN PARA FLUJO DE TRABAJO AUTOMÁTICO *** ---
def main():
    # 1. Definir nombres de carpetas
    INPUT_DIR_NAME = "audios"
    OUTPUT_DIR_NAME = "graficas"
    
    # 2. Obtener el directorio actual (donde se ejecuta el script)
    # Path.cwd() es el "Current Working Directory" (Directorio de Trabajo Actual)
    base_dir = Path.cwd()
    
    input_dir = base_dir / INPUT_DIR_NAME
    output_dir = base_dir / OUTPUT_DIR_NAME
    
    # 3. Crear carpetas y mostrar mensajes
    try:
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        print("--- 📂 Configuración de Carpetas ---")
        print(f"[OK] Carpeta de entrada lista: {input_dir.resolve()}")
        print(f"[OK] Carpeta de salida lista: {output_dir.resolve()}")
        print("-----------------------------------")
        
    except OSError as e:
        print(f"Error: No se pudieron crear las carpetas. ¿Permisos insuficientes?")
        print(f"Detalle: {e}")
        sys.exit(1) # Salir del script si hay un error
        
    # 4. Mensaje de espera para el usuario
    print("\n" + "="*50)
    print("  ¡ACCIÓN REQUERIDA! ✋")
    print(f"  Por favor, agrega tus archivos de audio (.wav, .mp3)")
    print(f"  a la carpeta: '{INPUT_DIR_NAME}'")
    print("="*50)
    
    # input() pausará el script hasta que el usuario presione Enter
    input("\n  Presiona Enter cuando estés listo para comenzar el análisis... ")
    print("\nIniciando análisis... 🚀")
    
    # 5. Iniciar el procesamiento (código del script anterior)
    patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
    files_found = 0
    
    for pat in patterns:
        for audio_path in input_dir.glob(pat):
            files_found += 1
            print(f"\n--- ({files_found}) Procesando: {audio_path.name} ---")
            
            # Nombre de archivo de salida (ej: "mi_nota.wav" -> "mi_nota_analisis.png")
            out_filename = f"{audio_path.stem}_analisis.png"
            output_path = output_dir / out_filename
            
            try:
                # Llamar a la función de análisis
                analyze_f0_and_harmonics(audio_path, output_path)
            except Exception as e:
                print(f"  > Error inesperado procesando {audio_path.name}: {e}")

    if files_found == 0:
        print(f"\nNo se encontraron archivos de audio {patterns} en la carpeta '{INPUT_DIR_NAME}'.")
        print("Asegúrate de haber copiado los archivos ANTES de presionar Enter.")
    else:
        print(f"\n¡Proceso completado! ✨ Se analizaron {files_found} archivos.")
        print(f"Revisa los resultados en la carpeta: '{OUTPUT_DIR_NAME}'")

if __name__ == "__main__":
    main()