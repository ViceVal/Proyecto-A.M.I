import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import collections

# --- CONFIGURACIÓN ---
FRECUENCIA_MUESTREO = 44100
TAMANO_VENTANA = 4096
A4 = 440.0

# --- PARÁMETROS DE "SENSACIÓN WEB" ---
TIEMPO_HOLD = 60
MARGEN_DETECCION = 15.0   # Rango para detectar qué cuerda es
BUFFER_SIZE = 4
UMBRAL_ESTABILIDAD = 3.0

# AQUÍ ESTÁ LA MAGIA: ¿Qué tan cerca debo estar para que se ponga verde?
# 1.5 Hz es un buen balance entre precisión y facilidad.
TOLERANCIA_AFINACION = 1.5 

# --- DICCIONARIO MEJORADO (Nombre + Código) ---
OBJETIVOS_GUITARRA = [
    {"nombre": "Mi",  "codigo": "E2", "freq": 82.41},  # 6ta
    {"nombre": "La",  "codigo": "A2", "freq": 110.00}, # 5ta
    {"nombre": "Re",  "codigo": "D3", "freq": 146.83}, # 4ta
    {"nombre": "Sol", "codigo": "G3", "freq": 196.00}, # 3ra
    {"nombre": "Si",  "codigo": "B3", "freq": 246.94}, # 2da
    {"nombre": "Mi",  "codigo": "E4", "freq": 329.63}  # 1ra
]

def harmonic_product_spectrum(magnitude, freqs, n_harmonics=3):
    hps_spec = magnitude.copy()
    for h in range(2, n_harmonics + 1):
        decimated = magnitude[::h] 
        hps_spec[:len(decimated)] *= decimated
    return hps_spec

class AfinadorV10_EstiloWeb:
    def __init__(self, root):
        self.root = root
        self.root.title("Afinador V10: Estilo Web")
        self.root.geometry("700x900")
        
        self.escuchando = False
        self.app_running = True
        self.buffer_freq = collections.deque(maxlen=BUFFER_SIZE)
        self.stream = None
        self.data_queue = queue.Queue()
        
        self.hold_counter = 0
        self.last_valid_data = None 
        
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # --- INTERFAZ GRÁFICA ---
        frame_top = tk.Frame(root, bg="#222", pady=20)
        frame_top.pack(fill=tk.X)
        
        # ETIQUETA DE NOTA (GIGANTE)
        self.lbl_nota_nombre = tk.Label(frame_top, text="--", font=("Arial", 90, "bold"), fg="white", bg="#222")
        self.lbl_nota_nombre.pack()
        
        # ETIQUETA DE CÓDIGO (E2, A2...)
        self.lbl_nota_codigo = tk.Label(frame_top, text="", font=("Arial", 30, "bold"), fg="#00d4ff", bg="#222")
        self.lbl_nota_codigo.pack()

        # ETIQUETA DE FRECUENCIA EXACTA
        self.lbl_freq = tk.Label(frame_top, text="...", font=("Arial", 14), fg="#777", bg="#222")
        self.lbl_freq.pack(pady=10)

        # PANEL DE INSTRUCCIONES
        self.frame_instr = tk.Frame(root, bg="#f0f0f0", pady=20)
        self.frame_instr.pack(fill=tk.X)
        
        self.lbl_accion = tk.Label(self.frame_instr, text="LISTO", font=("Arial", 28, "bold"), fg="#ccc", bg="#f0f0f0")
        self.lbl_accion.pack()
        
        # AGUJA VISUAL
        self.canvas = tk.Canvas(root, width=600, height=120, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Dibujar zona verde (tolerancia visual)
        centro = 300
        ancho_verde = TOLERANCIA_AFINACION * 10 * 2 # Escala visual
        self.canvas.create_rectangle(centro - ancho_verde, 10, centro + ancho_verde, 90, fill="#e8f5e9", outline="") # Fondo verde suave
        self.canvas.create_line(centro, 10, centro, 90, width=4, fill="#444") # Línea central
        
        # Marcas de escala
        for i in range(-60, 61, 15):
            if i == 0: continue
            x = 300 + (i * 5)
            self.canvas.create_line(x, 40, x, 60, fill="#999", width=2)
            
        self.aguja = self.canvas.create_polygon(300, 20, 280, 60, 320, 60, fill="#ccc")

        # GRÁFICA ESPECTRAL
        self.frame_grafica = tk.Frame(root)
        self.frame_grafica.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(5, 2), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.ax.set_title("Señal en tiempo real", fontsize=8)
        self.ax.set_yticks([]) 
        self.ax.set_xlim(60, 400)
        self.line, = self.ax.plot([], [], color='#2196F3', lw=1.5)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_grafica)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
            self.reset_ui()
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
        raw_audio = indata[:, 0] * 5.0 
        
        ventana = np.hanning(len(raw_audio))
        fft_spec = np.abs(np.fft.rfft(raw_audio * ventana))
        freqs = np.fft.rfftfreq(len(raw_audio), 1 / FRECUENCIA_MUESTREO)
        
        hps = harmonic_product_spectrum(fft_spec, freqs, n_harmonics=3)
        hps[freqs < 65] = 0 
        
        idx_pico = np.argmax(hps)
        freq_detectada = freqs[idx_pico]
        
        if 0 < idx_pico < len(hps) - 1:
            alpha = hps[idx_pico-1]; beta = hps[idx_pico]; gamma = hps[idx_pico+1]
            if beta > 0:
                p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                freq_detectada += p * (FRECUENCIA_MUESTREO / TAMANO_VENTANA)
        
        # Anti-confusión Mi vs Si
        if 235 < freq_detectada < 260:
            idx_mi = int(82.41 / (FRECUENCIA_MUESTREO / TAMANO_VENTANA))
            energia_mi = np.sum(fft_spec[idx_mi-2 : idx_mi+3])
            idx_si = idx_pico
            energia_si = fft_spec[idx_si]
            if energia_mi > (energia_si * 0.15):
                freq_detectada = freq_detectada / 3.0
                # Recalcular fino
                idx_mi_aprox = int(freq_detectada / (FRECUENCIA_MUESTREO / TAMANO_VENTANA))
                start = max(0, idx_mi_aprox - 5)
                end = min(len(fft_spec), idx_mi_aprox + 5)
                local_peak = np.argmax(fft_spec[start:end]) + start
                freq_detectada = freqs[local_peak]

        max_val = np.max(fft_spec)
        grafica = fft_spec / max_val if max_val > 0 else fft_spec
        try: self.data_queue.put_nowait((freq_detectada, grafica))
        except: pass

    def actualizar_gui(self):
        if not self.app_running: return
        try:
            freq_raw = 0
            grafica = None
            while not self.data_queue.empty():
                freq_raw, grafica = self.data_queue.get_nowait()
            
            if self.escuchando:
                if grafica is not None:
                    x_data = np.linspace(0, FRECUENCIA_MUESTREO/2, len(grafica))
                    self.line.set_data(x_data, grafica)
                    self.canvas_plot.draw_idle()

                detected = False
                if freq_raw > 65:
                    self.buffer_freq.append(freq_raw)
                    if len(self.buffer_freq) == BUFFER_SIZE:
                        if np.std(self.buffer_freq) < UMBRAL_ESTABILIDAD:
                            freq_prom = np.mean(self.buffer_freq)
                            cuerda, dif = self.encontrar_cuerda(freq_prom)
                            if cuerda:
                                detected = True
                                self.last_valid_data = (cuerda, dif, freq_prom)
                                self.hold_counter = TIEMPO_HOLD

                if detected:
                    self.dibujar(self.last_valid_data, "live")
                elif self.hold_counter > 0 and self.last_valid_data:
                    self.hold_counter -= 1
                    self.dibujar(self.last_valid_data, "hold")
                elif self.hold_counter == 0:
                    self.reset_ui_visuals()

        except Exception as e: print(e)
        self.root.after(30, self.actualizar_gui)

    def dibujar(self, data, estado):
        cuerda, dif, freq = data
        
        # 1. NOMBRES Y CÓDIGOS
        color_txt = "white" if estado == "live" else "#777"
        color_code = "#00d4ff" if estado == "live" else "#005f73"
        
        self.lbl_nota_nombre.config(text=cuerda["nombre"], fg=color_txt) # "Mi"
        self.lbl_nota_codigo.config(text=f"({cuerda['codigo']})", fg=color_code) # "(E2)"
        self.lbl_freq.config(text=f"Detectado: {freq:.1f} Hz  |  Objetivo: {cuerda['freq']} Hz")

        # 2. ESTADO DE AFINACIÓN (ZONA VERDE)
        msg, col = "", ""
        
        # Usamos la variable TOLERANCIA_AFINACION (ej. 1.5 Hz)
        if abs(dif) <= TOLERANCIA_AFINACION:
            msg, col = "✨ ¡AFINADO! ✨", "#4CAF50" # Verde
        elif dif < 0:
            msg, col = "APRETAR ⤴", "#ff9800" # Naranja
        else:
            msg, col = "SOLTAR ⤵", "#e74c3c" # Rojo
        
        self.lbl_accion.config(text=msg, fg=col)

        # 3. MOVER AGUJA
        off = max(min(dif, 15), -15)
        x = 300 + (off * 10)
        self.canvas.coords(self.aguja, x, 20, x-20, 60, x+20, 60)
        self.canvas.itemconfig(self.aguja, fill=col)

    def reset_ui(self):
        self.reset_ui_visuals()
        self.buffer_freq.clear()
        self.hold_counter = 0
        self.last_valid_data = None
    
    def reset_ui_visuals(self):
        self.lbl_nota_nombre.config(text="--", fg="white")
        self.lbl_nota_codigo.config(text="")
        self.lbl_freq.config(text="...")
        self.lbl_accion.config(text="LISTO", fg="#ccc")
        self.canvas.itemconfig(self.aguja, fill="#ccc")

if __name__ == "__main__":
    root = tk.Tk()
    app = AfinadorV10_EstiloWeb(root)
    root.mainloop()