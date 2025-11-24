# ============================================================
# core/efectos.py
# DSP – Pedalera de efectos
# ============================================================

import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter, fftconvolve
import soundfile as sf

from typing import Optional

# En caso de que este módulo se use junto al ESP32:
try:
    from core_esp32_serial import enviar_lcd
except:
    def enviar_lcd(*args, **kwargs):
        pass


FRECUENCIA_MUESTREO = 44100


class ProcesadorEfectos:
    """
    Clase independiente para aplicar efectos DSP:
    - Filtro paso bajos
    - Distorsión
    - Modulación
    - Delay
    - Reverb
    """

    def __init__(self):
        self.samplerate = FRECUENCIA_MUESTREO
        self.audio_original: Optional[np.ndarray] = None
        self.audio_procesado: Optional[np.ndarray] = None
        self.grabacion_actual: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # GRABAR AUDIO
    # ------------------------------------------------------------
    def grabar_audio(self, duracion: float = 3.0) -> np.ndarray:
        print(f"Grabando {duracion} segundos...")

        self.grabacion_actual = sd.rec(
            int(duracion * self.samplerate),
            samplerate=self.samplerate,
            channels=1
        )
        sd.wait()

        if self.grabacion_actual is not None:
            self.audio_original = self.grabacion_actual.flatten()

            # Normalizar
            if np.max(np.abs(self.audio_original)) > 0:
                self.audio_original = self.audio_original / np.max(np.abs(self.audio_original))

            print("Grabación completada.")

            enviar_lcd("E", "Grabado 3s", "Listo para FX", "", "")

            return self.audio_original

        return np.array([])

    # ------------------------------------------------------------
    # EFECTOS DSP
    # ------------------------------------------------------------
    def aplicar_efecto(self, efecto: str, parametro: float) -> np.ndarray:
        if self.audio_original is None:
            raise ValueError("No hay audio grabado para procesar")

        audio = self.audio_original.copy()
        val = parametro / 100.0
        resultado = audio

        # --- FILTRO PASA BAJOS ---
        if "Pasa-Bajos" in efecto:
            cutoff = 500 + (val * 4000)
            nyq = 0.5 * self.samplerate
            normal_cutoff = cutoff / nyq

            b, a = butter(5, normal_cutoff, btype='low')
            resultado = lfilter(b, a, audio)

        # --- DISTORSIÓN ---
        elif "Distorsión" in efecto:
            drive = 1 + (val * 20)
            resultado = np.tanh(audio * drive)
            resultado /= np.max(np.abs(resultado))

        # --- MODULACIÓN ---
        elif "Modulación" in efecto:
            freq_mod = 5 + (val * 500)
            t = np.arange(len(audio)) / self.samplerate
            carrier = np.sin(2 * np.pi * freq_mod * t)
            resultado = audio * 0.5 + (audio * carrier) * 0.5

        # --- DELAY ---
        elif "Delay" in efecto:
            delay_sec = 0.1 + (val * 0.8)
            delay_samples = int(delay_sec * self.samplerate)
            decay = 0.5

            output = np.zeros(len(audio) + delay_samples)
            output[:len(audio)] += audio
            output[delay_samples:] += audio * decay

            resultado = output

        # --- REVERB ---
        elif "Reverb" in efecto:
            reverb_len = int(self.samplerate * (0.5 + val * 2.0))
            t = np.linspace(0, 1, reverb_len)

            ir = np.random.randn(reverb_len) * np.exp(-5 * t)
            resultado = fftconvolve(audio, ir, mode='full')
            resultado /= np.max(np.abs(resultado)) * 0.9

        self.audio_procesado = resultado.astype(np.float64)

        enviar_lcd("E", "Efecto aplicado", efecto, f"Int {int(parametro)}%", "")

        return self.audio_procesado

    # ------------------------------------------------------------
    # REPRODUCIR
    # ------------------------------------------------------------
    def reproducir_original(self):
        if self.audio_original is not None:
            sd.stop()
            sd.play(self.audio_original, self.samplerate)

    def reproducir_procesado(self):
        if self.audio_procesado is not None:
            sd.stop()
            sd.play(self.audio_procesado, self.samplerate)

    # ------------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------------
    def guardar_audio(self, archivo: str, audio_data: np.ndarray):
        if audio_data is not None:
            sf.write(archivo, audio_data, self.samplerate)
