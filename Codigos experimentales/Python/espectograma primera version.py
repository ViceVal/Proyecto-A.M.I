import threading
import time
from dataclasses import dataclass
import math

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_NAMES_ES = {
    "C":  "Do",
    "C#": "Do#",
    "D":  "Re",
    "D#": "Re#",
    "E":  "Mi",
    "F":  "Fa",
    "F#": "Fa#",
    "G":  "Sol",
    "G#": "Sol#",
    "A":  "La",
    "A#": "La#",
    "B":  "Si"
}

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

def estimate_f0_yin(y, sr, fmin=60.0, fmax=1000.0, frame_length=2048, hop_length=256):
    """Devuelve la última f0 válida (Hz) usando librosa.yin; None si no hay."""
    try:
        f0_series = librosa.yin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length,
            trough_threshold=0.1
        )
        finite = np.isfinite(f0_series)
        if not np.any(finite):
            return None
        return float(f0_series[finite][-1])
    except Exception:
        return None

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
    device: int | None = None
    detect_chord: bool = False
    show_f0: bool = False
    f0_min: float = 60.0
    f0_max: float = 1000.0

class LiveSpectrogramApp:
    def __init__(self, master):
        self.master = master
        master.title("Espectrograma en tiempo real (Thonny)")

        if sd is None or librosa is None:
            messagebox.showerror(
                "Error",
                "Faltan librerías.\nInstala primero:\n\npip install sounddevice librosa matplotlib numpy soundfile"
            )

        self.cfg = RTConfig()

        self.running = False
        self.stream = None
        self.ring = RingBuffer(self.cfg.sr * self.cfg.seconds)

        self.create_widgets()
        self.create_plot()

        self.master.after(50, self.update_plot)

    def create_widgets(self):
        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_start = ttk.Button(top, text="Start", command=self.start_stream)
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.var_show_f0 = tk.BooleanVar(value=False)
        self.var_detect_chord = tk.BooleanVar(value=False)
        self.var_mel = tk.BooleanVar(value=False)

        ttk.Checkbutton(top, text="Mostrar f0", variable=self.var_show_f0).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(top, text="Detectar acorde", variable=self.var_detect_chord).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(top, text="Mel-espectrograma", variable=self.var_mel).pack(side=tk.LEFT, padx=5)

        self.label_status = ttk.Label(self.master, text="Listo.")
        self.label_status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def create_plot(self):
        frame_plot = ttk.Frame(self.master)
        frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Espectrograma en tiempo real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Frecuencia")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.img = None  

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        x = indata.astype(np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        self.ring.write(x)

    def start_stream(self):
        if self.running:
            return
        if sd is None or librosa is None:
            messagebox.showerror("Error", "Faltan librerías sounddevice/librosa.")
            return

        self.cfg.detect_chord = self.var_detect_chord.get()
        self.cfg.show_f0 = self.var_show_f0.get()
        self.cfg.mel = self.var_mel.get()

        try:
            self.stream = sd.InputStream(
                samplerate=self.cfg.sr,
                blocksize=self.cfg.block,
                channels=self.cfg.channels,
                device=self.cfg.device,
                callback=self.audio_callback
            )
            self.stream.start()
            self.running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.label_status.config(text="Capturando audio...")
        except Exception as e:
            messagebox.showerror("Error de audio", f"No se pudo iniciar el stream:\n{e}")
            self.running = False

    def stop_stream(self):
        if not self.running:
            return
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.label_status.config(text="Detenido.")

    def update_plot(self):
        if self.running:
            n = self.cfg.sr * self.cfg.seconds
            y = self.ring.read(n)

            if self.cfg.mel:
                S = librosa.feature.melspectrogram(
                    y=y, sr=self.cfg.sr,
                    n_fft=self.cfg.n_fft,
                    hop_length=self.cfg.hop,
                    n_mels=128,
                    fmax=self.cfg.fmax
                )
                S_db = librosa.power_to_db(S, ref=np.max)
                y_axis = "mel"
            else:
                S = np.abs(librosa.stft(
                    y, n_fft=self.cfg.n_fft,
                    hop_length=self.cfg.hop
                ))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                y_axis = "hz"

            self.ax.clear()
            librosa.display.specshow(
                S_db, sr=self.cfg.sr,
                hop_length=self.cfg.hop,
                x_axis="time", y_axis=y_axis,
                ax=self.ax
            )

            title_extra = ""

            if self.cfg.detect_chord:
                chroma = librosa.feature.chroma_cqt(y=y, sr=self.cfg.sr)
                chroma_mean = np.mean(chroma, axis=1)
                label, qual, score = detect_chord_from_chroma(chroma_mean, qualities=('maj','min'))
                if label is not None:
                    root_str = label[0]
                    if len(label) >= 2 and label[1] == "#":
                        root_str = label[:2]

                    nombre_es = NOTE_NAMES_ES.get(root_str, root_str)
                    tipo = "mayor" if qual == "maj" else "menor"

                    pc_root = NOTE_NAMES_SHARP.index(root_str)
                    if qual == "maj":
                        pcs = [pc_root, (pc_root + 4) % 12, (pc_root + 7) % 12] 
                    else:
                        pcs = [pc_root, (pc_root + 3) % 12, (pc_root + 7) % 12] 

                    notas_en = [NOTE_NAMES_SHARP[pc] for pc in pcs] 
                    notas_es = [NOTE_NAMES_ES[n] for n in notas_en] 

                    title_extra += f" · Acorde: {label} ({nombre_es} {tipo}, conf {score:.2f})"
                    self.label_status.config(
                        text=f"Acorde: {label} ({nombre_es} {tipo}) · "
                             f"Notas: {'-'.join(notas_en)} ({'-'.join(notas_es)})"
                    )

            if self.cfg.show_f0:
                f0 = estimate_f0_yin(
                    y, self.cfg.sr,
                    fmin=self.cfg.f0_min,
                    fmax=self.cfg.f0_max,
                    frame_length=max(2048, self.cfg.n_fft),
                    hop_length=max(128, self.cfg.hop // 2)
                )
                if f0 is not None:
                    note = hz_to_note_name(f0)
                    if note is None:
                        note_text = "?"
                    else:
                        note_text = note
                    title_extra += f" · f0 ≈ {f0:.1f} Hz ({note_text})"

            self.ax.set_title("Espectrograma en tiempo real" + title_extra)

            self.fig.tight_layout()
            self.canvas.draw()

        self.master.after(50, self.update_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveSpectrogramApp(root)
    root.mainloop()
