import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import soundfile as sf
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class PedaleraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedalera DSP: Procesador de Audio")
        self.root.geometry("900x700")
        
        # Variables de Audio
        self.filepath = ""
        self.data_original = None
        self.data_procesada = None
        self.samplerate = 44100
        
        # --- INTERFAZ ---
        
        # 1. Panel de Carga
        frame_top = tk.Frame(root, bg="#333", pady=10)
        frame_top.pack(fill=tk.X)
        
        self.btn_load = tk.Button(frame_top, text="📂 Cargar Archivo (.wav)", command=self.cargar_archivo, 
                                  bg="#007acc", fg="white", font=("Arial", 10, "bold"))
        self.btn_load.pack(side=tk.LEFT, padx=20)
        
        self.lbl_file = tk.Label(frame_top, text="Ningún archivo cargado", fg="#ccc", bg="#333")
        self.lbl_file.pack(side=tk.LEFT)

        # 2. Panel de Efectos
        frame_fx = tk.Frame(root, pady=10, bg="#f0f0f0")
        frame_fx.pack(fill=tk.X)
        
        tk.Label(frame_fx, text="Selecciona Efecto:", bg="#f0f0f0").grid(row=0, column=0, padx=10)
        
        self.efecto_var = tk.StringVar()
        opciones = ["Filtro Pasa-Bajos (Low Pass)", "Distorsión (Overdrive)", "Modulación (Ring Mod)", "Delay (Eco)", "Reverb (Convolución)"]
        self.combo_fx = ttk.Combobox(frame_fx, textvariable=self.efecto_var, values=opciones, state="readonly", width=30)
        self.combo_fx.current(0)
        self.combo_fx.grid(row=0, column=1, padx=10)
        
        tk.Label(frame_fx, text="Intensidad / Mix:", bg="#f0f0f0").grid(row=0, column=2, padx=10)
        self.slider_param = tk.Scale(frame_fx, from_=0, to=100, orient=tk.HORIZONTAL, length=200, bg="#f0f0f0")
        self.slider_param.set(50)
        self.slider_param.grid(row=0, column=3, padx=10)
        
        self.btn_process = tk.Button(frame_fx, text="⚡ PROCESAR", command=self.aplicar_efecto, bg="#e74c3c", fg="white", font=("bold"))
        self.btn_process.grid(row=0, column=4, padx=20)

        # 3. Gráficas
        self.frame_plot = tk.Frame(root)
        self.frame_plot.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 4))
        self.fig.tight_layout(pad=3.0)
        
        self.ax1.set_title("Original")
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Procesado (Con Efecto)")
        self.ax2.set_ylim(-1, 1)
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Controles de Reproducción
        frame_play = tk.Frame(root, bg="#ddd", pady=15)
        frame_play.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Button(frame_play, text="▶ Reproducir Original", command=lambda: self.reproducir(self.data_original), width=20).pack(side=tk.LEFT, padx=20)
        tk.Button(frame_play, text="▶ Reproducir EFECTO", command=lambda: self.reproducir(self.data_procesada), bg="#4CAF50", fg="white", width=20).pack(side=tk.LEFT, padx=20)
        tk.Button(frame_play, text="💾 Guardar Resultado", command=self.guardar_archivo, bg="#333", fg="white").pack(side=tk.RIGHT, padx=20)
        tk.Button(frame_play, text="⏹ Stop", command=sd.stop, bg="#e74c3c", fg="white").pack(side=tk.LEFT, padx=20)

    def cargar_archivo(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac")])
        if path:
            self.filepath = path
            self.lbl_file.config(text=os.path.basename(path))
            # Cargar audio
            data, fs = sf.read(path)
            self.samplerate = fs
            
            # Si es estéreo, convertir a mono para simplificar efectos
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Normalizar -1 a 1
            self.data_original = data / np.max(np.abs(data))
            self.data_procesada = self.data_original.copy()
            
            self.actualizar_grafica(self.ax1, self.data_original, "Original")
            self.actualizar_grafica(self.ax2, self.data_procesada, "Sin procesar")

    def actualizar_grafica(self, ax, data, title):
        ax.clear()
        ax.set_title(title)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if data is not None:
            # Submuestreo para graficar rápido (cada 100 muestras)
            ax.plot(data[::100], color='#007acc', lw=0.5)
        self.canvas.draw()

    def reproducir(self, data):
        if data is not None:
            sd.stop()
            sd.play(data, self.samplerate)

    def guardar_archivo(self):
        if self.data_procesada is None: return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV file", "*.wav")])
        if path:
            sf.write(path, self.data_procesada, self.samplerate)
            messagebox.showinfo("Éxito", "Archivo guardado correctamente.")

    # --- LÓGICA DE EFECTOS (DSP) ---

    def aplicar_efecto(self):
        if self.data_original is None:
            messagebox.showerror("Error", "Carga un archivo primero")
            return
            
        audio = self.data_original.copy()
        fs = self.samplerate
        val = self.slider_param.get() / 100.0 # 0.0 a 1.0
        efecto = self.efecto_var.get()
        
        resultado = audio

        if "Pasa-Bajos" in efecto:
            # Filtro Butterworth
            cutoff = 500 + (val * 4000) # De 500Hz a 4500Hz
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
            resultado = signal.lfilter(b, a, audio)

        elif "Distorsión" in efecto:
            # Hard Clipping / Tanh saturation
            drive = 1 + (val * 20) # Ganancia de 1x a 20x
            resultado = np.tanh(audio * drive)
            # Normalizar para que no reviente el volumen
            resultado = resultado / np.max(np.abs(resultado))

        elif "Modulación" in efecto:
            # Ring Modulator (Multiplicar por una onda seno)
            freq_mod = 5 + (val * 500) # De 5Hz (Tremolo) a 500Hz (Robot)
            t = np.arange(len(audio)) / fs
            carrier = np.sin(2 * np.pi * freq_mod * t)
            # Mix 50/50
            resultado = (audio * 0.5) + ((audio * carrier) * 0.5)

        elif "Delay" in efecto:
            # Eco simple
            delay_sec = 0.1 + (val * 0.8) # 0.1s a 0.9s
            delay_samples = int(delay_sec * fs)
            decay = 0.5
            
            # Crear array vacío más grande
            output = np.zeros(len(audio) + delay_samples)
            output[:len(audio)] += audio
            # Sumar copia retardada
            output[delay_samples:] += audio * decay
            resultado = output

        elif "Reverb" in efecto:
            # Convolución con ruido blanco (Simulación básica de sala)
            reverb_len = int(fs * (0.5 + val * 2.0)) # 0.5s a 2.5s de cola
            # Respuesta al impulso (Impulse Response)
            t = np.linspace(0, 1, reverb_len)
            ir = np.random.randn(reverb_len) * np.exp(-5 * t) # Ruido que decae exponencialmente
            
            # Convolución rápida (FFT)
            resultado = signal.fftconvolve(audio, ir, mode='full')
            # Normalizar
            resultado = resultado / np.max(np.abs(resultado)) * 0.9

        # Guardar y graficar
        self.data_procesada = resultado
        self.actualizar_grafica(self.ax2, self.data_procesada, f"Efecto: {efecto}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PedaleraApp(root)
    root.mainloop()