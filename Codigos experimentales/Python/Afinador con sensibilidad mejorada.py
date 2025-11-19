import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter
import queue
import collections

# --- CONFIGURACIÓN ---
FRECUENCIA_MUESTREO = 44100
TAMANO_VENTANA = 4096  # Reduje un poco para que sea más rápido refrescando
A4 = 440.0

# --- LISTA DE CUERDAS (VIP) ---
OBJETIVOS_GUITARRA = [
    {"nota": "Mi 2 (6ta)", "freq": 82.41},
    {"nota": "La 2 (5ta)", "freq": 110.00},
    {"nota": "Re 3 (4ta)", "freq": 146.83},
    {"nota": "Sol 3 (3ra)", "freq": 196.00},
    {"nota": "Si 3 (2da)", "freq": 246.94},
    {"nota": "Mi 4 (1ra)", "freq": 329.63}
]

# --- AJUSTES DE SENSIBILIDAD ---
MARGEN_DETECCION = 15.0   # Más permisivo (antes 10) para cuerdas desafinadas
BUFFER_SIZE = 4           # Menos lecturas necesarias para "fijar" la nota (más rápido)
UMBRAL_ESTABILIDAD = 3.0  # Más tolerante con pequeñas vibraciones

def harmonic_product_spectrum(magnitude, freqs, n_harmonics=4): #se puede cambiar el para que detecte mejor las notas.
    hps_spec = magnitude.copy()
    for h in range(2, n_harmonics + 1):
        decimated = magnitude[::h] 
        hps_spec[:len(decimated)] *= decimated
    return hps_spec

class AfinadorGuitarraV7:
    def __init__(self, root):
        self.root = root
        self.root.title("Afinador V7: Gráfica Viva + Mejor Detección")
        self.root.geometry("700x850")
        
        self.escuchando = False
        self.app_running = True
        self.buffer_freq = collections.deque(maxlen=BUFFER_SIZE)
        self.stream = None
        self.data_queue = queue.Queue()
        
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # --- UI ---
        frame_top = tk.Frame(root, bg="#1e1e1e", pady=20)
        frame_top.pack(fill=tk.X)
        
        tk.Label(frame_top, text="GUITARRA ESTÁNDAR", font=("Arial", 10, "bold"), fg="#00d4ff", bg="#1e1e1e").pack()
        self.lbl_nota = tk.Label(frame_top, text="--", font=("Arial", 90, "bold"), fg="white", bg="#1e1e1e")
        self.lbl_nota.pack()
        self.lbl_freq = tk.Label(frame_top, text="Silencio", font=("Arial", 14), fg="#aaa", bg="#1e1e1e")
        self.lbl_freq.pack(pady=5)
        self.lbl_estado = tk.Label(frame_top, text="...", font=("Arial", 10), fg="#777", bg="#1e1e1e")
        self.lbl_estado.pack()

        # Canvas Aguja
        self.canvas = tk.Canvas(root, width=600, height=120, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(pady=15)
        # Dibujar escala
        self.canvas.create_line(300, 10, 300, 110, width=4, fill="#444") # Centro
        self.canvas.create_text(300, 115, text="✅", fill="#444", font=("Arial", 12))
        for i in range(-50, 51, 10):
            if i == 0: continue
            x = 300 + (i * 5) # Escala visual
            h = 20 if i % 20 == 0 else 10
            self.canvas.create_line(x, 50, x, 50+h, fill="#999", width=2)
        self.aguja = self.canvas.create_polygon(300, 20, 285, 60, 315, 60, fill="#ccc")

        # Gráfica Espectro (SIEMPRE VISIBLE)
        self.frame_grafica = tk.Frame(root)
        self.frame_grafica.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 2.5), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_title("Señal de Entrada (Micrófono)")
        self.ax.set_xlabel("Frecuencia (Hz)")
        self.ax.set_yticks([]) 
        self.ax.set_xlim(60, 500) # Zoom en zona de guitarra
        
        # Línea espectro
        self.line, = self.ax.plot([], [], color='#007acc', lw=1.5)
        self.fill = self.ax.fill_between([], 0, 0, color='#007acc', alpha=0.2)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_grafica)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Botón
        self.btn_start = tk.Button(root, text="🎙️ ENCENDER", font=("Arial", 14, "bold"), 
                                   bg="#4CAF50", fg="white", command=self.toggle_audio, height=2)
        self.btn_start.pack(fill=tk.X, side=tk.BOTTOM, padx=30, pady=20)

        self.actualizar_gui()

    def cerrar_aplicacion(self):
        self.app_running = False
        if self.stream: self.stream.stop(); self.stream.close()
        self.root.destroy()

    def toggle_audio(self):
        if self.escuchando:
            self.escuchando = False
            self.btn_start.config(text="🎙️ ENCENDER", bg="#4CAF50")
            self.lbl_nota.config(text="--")
            if self.stream: self.stream.stop(); self.stream.close()
        else:
            self.escuchando = True
            self.btn_start.config(text="🛑 APAGAR", bg="#e74c3c")
            self.buffer_freq.clear()
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, 
                                         samplerate=FRECUENCIA_MUESTREO, blocksize=TAMANO_VENTANA)
            self.stream.start()

    def encontrar_cuerda(self, freq):
        mejor = None
        menor_dif = float('inf')
        for cuerda in OBJETIVOS_GUITARRA:
            dif = abs(freq - cuerda["freq"])
            if dif < menor_dif:
                menor_dif = dif
                mejor = cuerda
        
        if menor_dif <= MARGEN_DETECCION:
            return mejor, freq - mejor["freq"]
        return None, 0

    def audio_callback(self, indata, frames, time_info, status):
        if not self.escuchando: return
        raw_audio = indata[:, 0]
        
        # 1. Amplificación por software (Ayuda si el mic es bajo)
        raw_audio = raw_audio * 2.0 

        # 2. FFT & HPS
        ventana = np.hanning(len(raw_audio))
        fft_spec = np.abs(np.fft.rfft(raw_audio * ventana))
        freqs = np.fft.rfftfreq(len(raw_audio), 1 / FRECUENCIA_MUESTREO)
        
        # Usamos n_harmonics=2 para salvar el La y el Re
        hps = harmonic_product_spectrum(fft_spec, freqs, n_harmonics=2)
        
        # Limpieza de graves extremos
        hps[freqs < 65] = 0
        
        idx_pico = np.argmax(hps)
        freq_detectada = freqs[idx_pico]
        
        # Interpolación
        if 0 < idx_pico < len(hps) - 1:
            alpha = hps[idx_pico-1]
            beta = hps[idx_pico]
            gamma = hps[idx_pico+1]
            if beta > 0:
                p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                freq_detectada += p * (FRECUENCIA_MUESTREO / TAMANO_VENTANA)
        
        # Preparar datos para gráfica (Normalizados)
        max_val = np.max(fft_spec) # Usamos el espectro puro para la gráfica, más bonito
        grafica = fft_spec / max_val if max_val > 0 else fft_spec
        
        try:
            self.data_queue.put_nowait((freq_detectada, grafica))
        except: pass

    def actualizar_gui(self):
        if not self.app_running: return
        try:
            freq_raw = 0
            grafica = None
            
            # Consumir cola
            while not self.data_queue.empty():
                freq_raw, grafica = self.data_queue.get_nowait()
            
            if self.escuchando:
                # 1. ACTUALIZAR GRÁFICA (Siempre, detecte nota o no)
                if grafica is not None:
                    x_data = np.linspace(0, FRECUENCIA_MUESTREO/2, len(grafica))
                    # Truco: Cortamos visualmente para que coincida con xlim(60, 500)
                    # Esto hace la gráfica muy fluida
                    self.line.set_data(x_data, grafica)
                    self.canvas_plot.draw_idle()

                # 2. PROCESAR NOTA
                if freq_raw > 65:
                    self.buffer_freq.append(freq_raw)
                
                es_estable = False
                freq_promedio = 0
                
                # Solo analizamos si el buffer está lleno
                if len(self.buffer_freq) == BUFFER_SIZE:
                    if np.std(self.buffer_freq) < UMBRAL_ESTABILIDAD:
                        es_estable = True
                        freq_promedio = np.mean(self.buffer_freq)
                
                if es_estable:
                    cuerda, dif = self.encontrar_cuerda(freq_promedio)
                    if cuerda:
                        self.lbl_nota.config(text=cuerda["nota"].split()[0], fg="#4CAF50")
                        self.lbl_freq.config(text=f"{freq_promedio:.1f} Hz")
                        self.lbl_estado.config(text=f"Afinando {cuerda['nota']}", fg="#fff")
                        
                        # Movimiento aguja
                        offset = max(min(dif, 15), -15) # Limite visual +-15Hz
                        x_pos = 300 + (offset * 10) # Factor escala visual
                        
                        color = "#4CAF50" if abs(dif) < 0.8 else "#ff9800" # Verde si <0.8Hz error
                        if abs(dif) > 5: color = "#e74c3c"
                        
                        self.canvas.coords(self.aguja, x_pos, 20, x_pos-15, 60, x_pos+15, 60)
                        self.canvas.itemconfig(self.aguja, fill=color)
                    else:
                        # Estable pero no es guitarra (ruido constante)
                        self.lbl_estado.config(text="Señal ignorada (No es guitarra)", fg="#555")
                else:
                    # Inestable
                    if len(self.buffer_freq) > 0:
                        self.lbl_estado.config(text="Escuchando...", fg="#555")
                    
                    # Opcional: Regresar aguja al centro o dejarla
                    # self.canvas.itemconfig(self.aguja, fill="#ccc")

        except Exception as e: print(e)
        self.root.after(30, self.actualizar_gui)

if __name__ == "__main__":
    root = tk.Tk()
    app = AfinadorGuitarraV7(root)
    root.mainloop()