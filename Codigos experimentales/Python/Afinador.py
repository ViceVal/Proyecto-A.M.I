import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter
import threading
import queue
import time

# --- CONFIGURACIÓN DE AUDIO ---
FRECUENCIA_MUESTREO = 44100
TAMANO_VENTANA = 4096 
A4 = 440.0

# --- FILTRO ---
LOW_CUT = 70.0
HIGH_CUT = 1200.0

# --- CONFIGURACIÓN DE MEMORIA VISUAL ---
TIEMPO_MEMORIA = 40  # Cantidad de "frames" que la pantalla se congela (aprox 2 segundos)

REFERENCIA_GUITARRA = {
    "Mi 2": 82.41, "La 2": 110.00, "Re 3": 146.83,
    "Sol 3": 196.00, "Si 3": 246.94, "Mi 4": 329.63
}
NOTAS = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class AfinadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Afinador Estable (Con Memoria)")
        self.root.geometry("700x780")
        
        self.escuchando = False
        self.app_running = True
        self.mostrar_grafica = tk.BooleanVar(value=True)
        self.valor_gate = 1.5
        
        # VARIABLES DE ESTABILIDAD (NUEVO)
        self.contador_silencio = 0
        self.ultimo_estado_valido = None # Guardará (nombre, dif, freq)
        
        self.stream = None
        self.data_queue = queue.Queue()
        
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # --- INTERFAZ GRÁFICA ---
        
        # 1. Panel Superior
        frame_info = tk.Frame(root, pady=10)
        frame_info.pack(fill=tk.X)
        
        self.lbl_nota = tk.Label(frame_info, text="--", font=("Arial", 60, "bold"), fg="#333")
        self.lbl_nota.pack()
        
        self.lbl_freq = tk.Label(frame_info, text="0.00 Hz", font=("Arial", 14), fg="#666")
        self.lbl_freq.pack()
        
        self.lbl_instruccion = tk.Label(frame_info, text="Listo para afinar", font=("Arial", 16, "bold"), fg="#007acc")
        self.lbl_instruccion.pack(pady=5)

        # 2. Barra de Afinación
        self.canvas_afinador = tk.Canvas(root, width=500, height=60, bg="#2b2b2b")
        self.canvas_afinador.pack(pady=10)
        self.canvas_afinador.create_line(250, 0, 250, 60, width=2, fill="white") # Centro
        self.canvas_afinador.create_text(250, 55, text="▼", fill="white", font=("Arial", 10))
        self.canvas_afinador.create_line(125, 15, 125, 45, width=1, fill="gray") 
        self.canvas_afinador.create_line(375, 15, 375, 45, width=1, fill="gray")
        
        # Aguja triangular
        self.aguja = self.canvas_afinador.create_polygon(250, 10, 240, 40, 260, 40, fill="white")

        # 3. Gráfica
        self.frame_grafica = tk.Frame(root)
        self.frame_grafica.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_title("Espectro (Hold activo)")
        self.ax.set_xlabel("Frecuencia (Hz)")
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(60, 800) # Zoom optimizado para guitarra
        self.ax.grid(True, alpha=0.3)
        
        self.x_data = np.linspace(0, FRECUENCIA_MUESTREO/2, TAMANO_VENTANA//2 + 1)
        self.line, = self.ax.plot([], [], color='#e91e63', lw=1.5)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_grafica)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toggle_grafica()

        # 4. Controles
        frame_controles = tk.Frame(root, bg="#ddd", pady=15, bd=1, relief=tk.RAISED)
        frame_controles.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.btn_accion = tk.Button(frame_controles, text="🎙️ INICIAR", font=("Arial", 11, "bold"), 
                                    command=self.toggle_audio, bg="#4CAF50", fg="white", width=12)
        self.btn_accion.pack(side=tk.LEFT, padx=20)

        frame_slider = tk.Frame(frame_controles, bg="#ddd")
        frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        tk.Label(frame_slider, text="🛡️ Sensibilidad (Gate)", bg="#ddd", font=("Arial", 9)).pack(anchor="w")
        self.slider_gate = tk.Scale(frame_slider, from_=0, to=60, orient=tk.HORIZONTAL, 
                                    bg="#ddd", highlightthickness=0, command=self.actualizar_gate_var)
        self.slider_gate.set(20) # Un poco más alto por defecto
        self.slider_gate.pack(fill=tk.X)

        self.chk_grafica = tk.Checkbutton(frame_controles, text="Ver Gráfica", 
                                          variable=self.mostrar_grafica, command=self.toggle_grafica, bg="#ddd")
        self.chk_grafica.pack(side=tk.RIGHT, padx=10)

        self.actualizar_gui()

    def cerrar_aplicacion(self):
        self.app_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.root.destroy()

    def actualizar_gate_var(self, val):
        try:
            self.valor_gate = float(val) / 10.0
        except:
            pass

    def toggle_grafica(self):
        if self.mostrar_grafica.get():
            self.frame_grafica.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        else:
            self.frame_grafica.pack_forget()

    def toggle_audio(self):
        if self.escuchando:
            self.escuchando = False
            self.btn_accion.config(text="🎙️ INICIAR", bg="#4CAF50")
            self.lbl_instruccion.config(text="Pausado", fg="#666")
            if self.stream:
                self.stream.stop()
                self.stream.close()
        else:
            self.escuchando = True
            self.btn_accion.config(text="🛑 PARAR", bg="#e74c3c")
            self.lbl_instruccion.config(text="Toca una nota...", fg="#333")
            self.contador_silencio = 0 # Resetear memoria
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, 
                                         samplerate=FRECUENCIA_MUESTREO, blocksize=TAMANO_VENTANA)
            self.stream.start()

    def audio_callback(self, indata, frames, time_info, status):
        if not self.escuchando: return
        
        raw_audio = indata[:, 0]
        
        # Filtro
        try:
            audio_data = aplicar_filtro(raw_audio, LOW_CUT, HIGH_CUT, FRECUENCIA_MUESTREO)
        except:
            audio_data = raw_audio
        
        # RMS (Volumen)
        volumen_rms = np.linalg.norm(audio_data)
        
        # --- LÓGICA DEL GATE MEJORADA ---
        # Si el volumen es bajo, mandamos señal de "Silencio" (None)
        if volumen_rms < self.valor_gate:
            try:
                self.data_queue.put_nowait(("SILENCIO", None, 0))
            except queue.Full: pass
            return

        # FFT
        ventana = np.hanning(len(audio_data))
        fft_spectrum = np.fft.rfft(audio_data * ventana)
        magnitud = np.abs(fft_spectrum)
        max_val = np.max(magnitud)
        magnitud_norm = magnitud / max_val if max_val > 0 else magnitud
        
        # Detección
        freqs = np.fft.rfftfreq(len(audio_data), 1 / FRECUENCIA_MUESTREO)
        magnitud[freqs < 60] = 0
        
        idx_pico = np.argmax(magnitud)
        freq_detectada = freqs[idx_pico]
        
        if 0 < idx_pico < len(magnitud) - 1:
            alpha = magnitud[idx_pico - 1]
            beta = magnitud[idx_pico]
            gamma = magnitud[idx_pico + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            freq_detectada += p * (FRECUENCIA_MUESTREO / TAMANO_VENTANA)
            
        try:
            self.data_queue.put_nowait((freq_detectada, magnitud_norm, volumen_rms))
        except queue.Full:
            pass

    def procesar_nota(self, freq):
        if freq == 0: return "--", 0, 0
        n = 12 * np.log2(freq / A4)
        n_round = int(round(n))
        freq_ideal = A4 * 2**(n_round/12)
        midi = n_round + 69
        nota = NOTAS[midi % 12]
        octava = (midi // 12) - 1
        return f"{nota} {octava}", freq_ideal, freq - freq_ideal

    def actualizar_gui(self):
        if not self.app_running: return

        try:
            # Procesar cola
            dato_nuevo = False
            tipo_dato = None # "SONIDO" o "SILENCIO"
            
            freq_temp = 0
            mag_temp = None
            
            # Vaciamos la cola para quedarnos con el último dato más reciente
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                dato_nuevo = True
                if item[0] == "SILENCIO":
                    tipo_dato = "SILENCIO"
                else:
                    tipo_dato = "SONIDO"
                    freq_temp, mag_temp, _ = item
            
            if self.escuchando and dato_nuevo:
                
                if tipo_dato == "SONIDO" and freq_temp > 60:
                    # --- CASO 1: SONIDO DETECTADO ---
                    self.contador_silencio = 0 # Resetear contador de apagado
                    
                    nombre, ideal, dif = self.procesar_nota(freq_temp)
                    
                    # Guardar este estado en memoria
                    self.ultimo_estado_valido = {
                        "nombre": nombre,
                        "ideal": ideal,
                        "freq": freq_temp,
                        "dif": dif,
                        "color": "#000"
                    }
                    
                    # ACTUALIZAR UI INMEDIATAMENTE
                    self.dibujar_tuner(nombre, freq_temp, ideal, dif)
                    
                    # Actualizar gráfica
                    if self.mostrar_grafica.get() and mag_temp is not None:
                        self.line.set_data(self.x_data[:len(mag_temp)], mag_temp)
                        self.canvas_plot.draw_idle()

                elif tipo_dato == "SILENCIO":
                    # --- CASO 2: SILENCIO (MEMORIA ACTIVA) ---
                    self.contador_silencio += 1
                    
                    # Si hace poco que hubo sonido (ej. menos de 2 segundos)
                    if self.contador_silencio < TIEMPO_MEMORIA and self.ultimo_estado_valido:
                        # MANTENER LA PANTALLA CONGELADA CON EL ÚLTIMO DATO
                        datos = self.ultimo_estado_valido
                        # Opcional: Cambiar color a gris para indicar que es "Memoria"
                        self.lbl_nota.config(fg="#888") 
                        self.lbl_instruccion.config(text="Manteniendo...", fg="#888")
                        # No borramos la aguja, la dejamos donde estaba
                    else:
                        # YA PASÓ MUCHO TIEMPO -> LIMPIAR PANTALLA
                        self.reset_ui()
                        if self.mostrar_grafica.get():
                            self.line.set_data([], [])
                            self.canvas_plot.draw_idle()

        except Exception as e:
            print(f"Error GUI: {e}")
        
        if self.app_running:
            self.root.after(50, self.actualizar_gui)

    def dibujar_tuner(self, nombre, freq, ideal, dif):
        es_cuerda = nombre in REFERENCIA_GUITARRA
        icono = "🎸" if es_cuerda else ""
        color_nota = "#000" if es_cuerda else "#555"
        
        self.lbl_nota.config(text=f"{icono} {nombre} {icono}", fg=color_nota)
        self.lbl_freq.config(text=f"{freq:.2f} Hz (Ideal: {ideal:.2f})")
        
        offset = max(min(dif, 2), -2) 
        pos_x = 250 + (offset * 50)
        
        color = "white"
        txt = ""
        
        if abs(dif) < 0.15:
            color = "#00cc00" # Verde
            txt = "✨ ¡PERFECTO! ✨"
        elif dif < 0:
            color = "#ff9800" # Naranja
            txt = "APRETAR (+)"
        else:
            color = "#ff3d00" # Rojo
            txt = "SOLTAR (-)"
        
        self.canvas_afinador.coords(self.aguja, pos_x, 10, pos_x-10, 40, pos_x+10, 40)
        self.canvas_afinador.itemconfig(self.aguja, fill=color)
        self.lbl_instruccion.config(text=txt, fg=color)

    def reset_ui(self):
        self.lbl_nota.config(text="--", fg="#333")
        self.lbl_freq.config(text="Silencio / Gate")
        self.lbl_instruccion.config(text="...", fg="gray")
        self.canvas_afinador.itemconfig(self.aguja, fill="gray")
        self.ultimo_estado_valido = None

if __name__ == "__main__":
    root = tk.Tk()
    app = AfinadorApp(root)
    root.mainloop()