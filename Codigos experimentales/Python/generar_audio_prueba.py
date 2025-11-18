import numpy as np
import librosa
import soundfile as sf  # soundfile es excelente para escribir archivos de audio

def generar_nota_sintetica(nota="A4", duracion_s=3.0, sr=22050, outfile="mi_nota_sintetica.wav"):
    """
    Genera una nota con armónicos y la guarda en un archivo WAV.
    """
    print(f"Generando nota: {nota} (duración {duracion_s}s) a {sr} Hz...")
    
    # 1. Obtener la frecuencia fundamental (f0) de la nota
    f0 = librosa.note_to_hz(nota)
    
    # 2. Crear el eje de tiempo
    t = np.linspace(0., duracion_s, int(sr * duracion_s), endpoint=False)
    
    # 3. Crear la forma de onda sumando armónicos
    y = np.zeros_like(t)
    
    # Amplitudes de los armónicos (la primera es f0, la segunda es 2*f0, etc.)
    # Esto le da el "timbre" al sonido
    amplitudes = [1.0, 0.7, 0.5, 0.3, 0.15] 
    
    for n, amp in enumerate(amplitudes, start=1):
        # Frecuencia del armónico actual (n=1 es f0)
        harmonic_freq = f0 * n
        
        # Generar la onda sinusoidal para este armónico
        sine_wave = np.sin(2 * np.pi * harmonic_freq * t)
        
        # Añadirlo a la señal total
        y += amp * sine_wave
        
    # 4. Aplicar un suave decaimiento (envelope) para que no sea un tono constante
    decay = np.exp(-t * 1.5)
    y *= decay
        
    # 5. Normalizar el audio para evitar clipping
    y /= np.max(np.abs(y))
    
    # 6. Guardar el archivo
    try:
        sf.write(outfile, y, sr)
        print(f"¡Éxito! Audio de prueba guardado en: {outfile}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
        print("Asegúrate de tener 'soundfile' instalado: pip install soundfile")

# --- Ejecutar el script ---
if __name__ == "__main__":
    # Puedes cambiar "A4" por "E2", "G3", etc.
    generar_nota_sintetica(nota="E2", duracion_s=3.0)