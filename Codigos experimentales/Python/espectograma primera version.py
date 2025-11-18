import argparse
import queue
import threading
import time
from dataclasses import dataclass
import math

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.argv = ["live_spectrogram.py", "--show_f0", "--detect_chord"]

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def major_template(root):
    tpl = np.zeros(12, dtype=float)
    tpl[root % 12] = 1.0
    tpl[(root + 4) % 12] = 0.8
    tpl[(root + 7) % 12] = 0.8
    return tpl

def minor_template(root):
    tpl = np.zeros(12, dtype=float)
    tpl[root % 12] = 1.0
    tpl[(root + 3) % 12] = 0.8
    tpl[(root + 7) % 12] = 0.8
    return tpl

def normalize_vec(v, eps=1e-8):
    s = np.linalg.norm(v) + eps
    return v / s

def detect_chord_from_chroma(chroma_mean, qualities=('maj','min')):
    chroma_n = normalize_vec(chroma_mean)
    best_score = -np.inf
    best = None
    for r in range(12):
        if 'maj' in qualities:
            tpl = normalize_vec(major_template(r))
            score = float(np.dot(chroma_n, tpl))
            if score > best_score:
                best_score = score
                best = (NOTE_NAMES_SHARP[r], 'maj', score)
        if 'min' in qualities:
            tpl = normalize_vec(minor_template(r))
            score = float(np.dot(chroma_n, tpl))
            if score > best_score:
                best_score = score
                best = (NOTE_NAMES_SHARP[r], 'min', score)
    if best is None:
        return None, None, 0.0
    root, qual, sc = best
    label = f"{root}{'' if qual=='maj' else 'm'}"
    return label, qual, float(sc)

def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(f / 440.0)

def midi_to_note_name(m):
    m_rounded = int(round(m))
    pc = m_rounded % 12
    octave = (m_rounded // 12) - 1
    return f"{NOTE_NAMES_SHARP[pc]}{octave}"

def hz_to_note_name(f):
    if f is None or f <= 0 or not np.isfinite(f):
        return None
    m = hz_to_midi(f)
    return midi_to_note_name(m)

@dataclass
class RTConfig:
    sr: int = 22050
    block: int = 1024
    hop: int = 512
    n_fft: int = 2048
    seconds: int = 5
    channels: int = 1
    mel: bool = False
    fmax: int = 8000
    device: str | None = None
    detect_chord: bool = False
    chord_qualities: tuple = ('maj','min')
    show_f0: bool = False
    f0_min: float = 60.0
    f0_max: float = 1000.0

class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.lock = threading.Lock()

    def write(self, data):
        n = len(data)
        with self.lock:
            idx = (self.pos + np.arange(n)) % self.size
            self.buf[idx] = data
            self.pos = (self.pos + n) % self.size

    def read(self, n):
        with self.lock:
            start = (self.pos - n) % self.size
            idx = (start + np.arange(n)) % self.size
            return self.buf[idx].copy()

def estimate_f0_yin(y, sr, fmin=60.0, fmax=1000.0, frame_length=2048, hop_length=256):
    try:
        f0_series = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=frame_length, hop_length=hop_length,
                                trough_threshold=0.1)
        finite = np.isfinite(f0_series)
        if not np.any(finite):
            return None
        return float(f0_series[finite][-1])
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Espectrograma en tiempo real desde el micrófono (+ detección de acorde y f0 opcionales).")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (Hz).")
    parser.add_argument("--block", type=int, default=1024, help="Tamaño de bloque de captura.")
    parser.add_argument("--channels", type=int, default=1, help="Número de canales de entrada (1=mono).")
    parser.add_argument("--n_fft", type=int, default=2048, help="Tamaño FFT para espectrograma.")
    parser.add_argument("--hop", type=int, default=512, help="Hop length para espectrograma.")
    parser.add_argument("--seconds", type=int, default=5, help="Ventana de visualización (s).")
    parser.add_argument("--mel", action="store_true", help="Usar mel-espectrograma.")
    parser.add_argument("--fmax", type=int, default=8000, help="Frecuencia máxima para mel.")
    parser.add_argument("--device", type=str, default=None, help="Nombre/ID de dispositivo de audio (opcional).")
    parser.add_argument("--detect_chord", action="store_true", help="Mostrar detección de acorde en tiempo real.")
    parser.add_argument("--show_f0", action="store_true", help="Mostrar frecuencia fundamental estimada y nota cercana.")
    parser.add_argument("--f0_min", type=float, default=60.0, help="f0 mínima para YIN (Hz).")
    parser.add_argument("--f0_max", type=float, default=1000.0, help="f0 máxima para YIN (Hz).")
    args = parser.parse_args()

    if sd is None:
        raise RuntimeError("Falta 'sounddevice'. Instala con: pip install sounddevice")
    if librosa is None:
        raise RuntimeError("Falta 'librosa'. Instala con: pip install librosa soundfile")

    cfg = RTConfig(sr=args.sr, block=args.block, hop=args.hop, n_fft=args.n_fft,
                   seconds=args.seconds, channels=args.channels, mel=args.mel,
                   fmax=args.fmax, device=args.device, detect_chord=args.detect_chord,
                   show_f0=args.show_f0, f0_min=args.f0_min, f0_max=args.f0_max)

    ring = RingBuffer(cfg.sr * cfg.seconds)
    q_err = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            q_err.put(status)
        x = indata.astype(np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        ring.write(x)

    stream = sd.InputStream(samplerate=cfg.sr, blocksize=cfg.block,
                            device=cfg.device, channels=cfg.channels,
                            callback=audio_cb)
    stream.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    img = None
    cbar = None
    last_title = ""

    try:
        while plt.fignum_exists(fig.number):
            if not q_err.empty():
                print("Audio status:", q_err.get_nowait())

            y = ring.read(cfg.sr * cfg.seconds)

            if cfg.mel:
                S = librosa.feature.melspectrogram(y=y, sr=cfg.sr, n_fft=cfg.n_fft,
                                                   hop_length=cfg.hop, n_mels=128, fmax=cfg.fmax)
                S_db = librosa.power_to_db(S, ref=np.max)
                y_axis = "mel"
            else:
                S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                y_axis = "hz"

            ax.cla()
            img = librosa.display.specshow(S_db, sr=cfg.sr, hop_length=cfg.hop, x_axis="time", y_axis=y_axis, ax=ax)
            if cbar is None:
                cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
                cbar.set_label("Intensidad (dB)")
            else:
                cbar.update_normal(img)

            title_extra = ""

            if cfg.detect_chord:
                chroma = librosa.feature.chroma_cqt(y=y, sr=cfg.sr)
                chroma_mean = np.mean(chroma, axis=1)
                label, qual, score = detect_chord_from_chroma(chroma_mean, qualities=('maj','min'))
                if label is not None:
                    title_extra += f" · Acorde: {label} (conf {score:.2f})"

            if cfg.show_f0:
                f0 = estimate_f0_yin(y, cfg.sr, fmin=cfg.f0_min, fmax=cfg.f0_max,
                                     frame_length=max(2048, cfg.n_fft), hop_length=max(128, cfg.hop//2))
                if f0 is not None:
                    note = hz_to_note_name(f0)
                    title_extra += f" · f0 ≈ {f0:.1f} Hz ({note})"

            title = f"Espectrograma en tiempo real{title_extra}"
            if title != last_title:
                ax.set_title(title)
                last_title = title

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
