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
from collections import deque
import wave
import os
from datetime import datetime

# --- CONFIGURACIÓN DE AUDIO ---
FRECUENCIA_MUESTREO = 44100
TAMANO_VENTANA = 4096 
A4 = 440.0

# --- FILTRO ---
LOW_CUT = 70.0
HIGH_CUT = 1200.0

# --- CONFIGURACIÓN DE MEMORIA VISUAL ---
TIEMPO_MEMORIA = 40

# --- SISTEMA DE CALIBRACIÓN ---
TIEMPO_CALIBRACION = 0.5
MUESTRAS_CALIBRACION = int(FRECUENCIA_MUESTREO * TIEMPO_CALIBRACION / TAMANO_VENTANA)

# --- DETECCIÓN DE ACORDES ---
HISTORIAL_NOTAS = 10
UMBRAL_ARMONICOS = 0.3

# --- GRABACIÓN ---
DIRECTORIO_GRABACIONES = "grabaciones_guitarra"
DURACION_MAXIMA_GRABACION = 30  # segundos

REFERENCIA_GUITARRA = {
    "Mi 2": 82.41, "La 2": 110.00, "Re 3": 146.83,
    "Sol 3": 196.00, "Si 3": 246.94, "Mi 4": 329.63
}

NOTAS = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

ACORDES = {
    "Do Mayor": ["Do", "Mi", "Sol"],
    "Do Menor": ["Do", "Re#", "Sol"],
    "Re Mayor": ["Re", "Fa#", "La"],
    "Re Menor": ["Re", "Fa", "La"],
    "Mi Mayor": ["Mi", "Sol#", "Si"],
    "Mi Menor": ["Mi", "Sol", "Si"],
    "Fa Mayor": ["Fa", "La", "Do"],
    "Fa Menor": ["Fa", "Sol#", "Do"],
    "Sol Mayor": ["Sol", "Si", "Re"],
    "Sol Menor": ["Sol", "La#", "Re"],
    "La Mayor": ["La", "Do#", "Mi"],
    "La Menor": ["La", "Do", "Mi"],
    "Si Mayor": ["Si", "Re#", "Sol#"],
    "Si Menor": ["Si", "Re", "Sol#"],
    "Do7": ["Do", "Mi", "Sol", "La#"],
    "Sol7": ["Sol", "Si", "Re", "Fa"],
    "La7": ["La", "Do#", "Mi", "Sol"],
    "Sus2": ["Do", "Re", "Sol"],
    "Sus4": ["Do", "Fa", "Sol"]
}

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

def nota_a_numero(nota):
    """Convierte nombre de nota a número (0-11)"""
    return NOTAS.index(nota.split()[0])

def detectar_acorde(notas_detectadas):
    """Detecta el acorde basado en las notas detectadas"""
    if len(notas_detectadas) < 2:
        return "Nota simple", 0
    
    numeros_notas = [nota_a_numero(nota) for nota in notas_detectadas if nota != "--"]
    
    if len(numeros_notas) < 2:
        return "Nota simple", 0
    
    fundamental_idx = min(numeros_notas)
    fundamental = NOTAS[fundamental_idx]
    
    intervalos = [(n - fundamental_idx) % 12 for n in numeros_notas]
    intervalos = sorted(set(intervalos))
    
    mejor_acorde = "Desconocido"
    mejor_puntaje = 0
    
    for nombre_acorde, notas_acorde in ACORDES.items():
        numeros_acorde = [nota_a_numero(nota) for nota in notas_acorde]
        intervalos_acorde = [(n - numeros_acorde[0]) % 12 for n in numeros_acorde]
        
        coincidencias = sum(1 for intervalo in intervalos if intervalo in intervalos_acorde)
        puntaje = coincidencias / len(intervalos_acorde)
        
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_acorde = nombre_acorde
    
    if mejor_puntaje > 0.6:
        return mejor_acorde, mejor_puntaje
    else:
        return f"{fundamental} (nota)", mejor_puntaje

class AfinadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Afinador Profesional + Detector de Acordes + Grabadora")
        self.root.geometry("850x1100")
        
        self.escuchando = False
        self.app_running = True
        self.mostrar_grafica = tk.BooleanVar(value=True)
        self.valor_gate = 1.5
        
        # --- VARIABLES DE CALIBRACIÓN ---
        self.nota_seleccionada = tk.StringVar(value="La 2")
        self.frecuencia_ideal = REFERENCIA_GUITARRA["La 2"]
        self.calibrando = False
        self.muestras_calibracion = []
        self.contador_calibracion = 0
        self.frecuencia_calibrada = None
        
        # --- DETECCIÓN DE ACORDES ---
        self.historico_notas = deque(maxlen=HISTORIAL_NOTAS)
        self.acorde_actual = "---"
        self.confianza_acorde = 0
        self.modo_acordes = False
        
        # --- GRABACIÓN (NUEVO) ---
        self.grabando = False
        self.audio_grabado = []
        self.tiempo_grabacion = 0
        self.duracion_grabacion = 30  # segundos por defecto
        
        # Variables de estabilidad
        self.contador_silencio = 0
        self.ultimo_estado_valido = None
        
        self.stream = None
        self.data_queue = queue.Queue()
        
        # Crear directorio de grabaciones
        if not os.path.exists(DIRECTORIO_GRABACIONES):
            os.makedirs(DIRECTORIO_GRABACIONES)
        
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.crear_interfaz()
        self.actualizar_gui()

    def crear_interfaz(self):
        # 1. Panel de Selección de Nota
        frame_seleccion = tk.Frame(self.root, bg="#e8f4fd", pady=10, relief=tk.RAISED, bd=1)
        frame_seleccion.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(frame_seleccion, text="🎸 SELECCIONAR CUERDA:", 
                font=("Arial", 12, "bold"), bg="#e8f4fd").pack(side=tk.LEFT, padx=10)
        
        self.combo_notas = ttk.Combobox(frame_seleccion, textvariable=self.nota_seleccionada,
                                       values=list(REFERENCIA_GUITARRA.keys()), 
                                       state="readonly", font=("Arial", 11), width=10)
        self.combo_notas.pack(side=tk.LEFT, padx=5)
        self.combo_notas.bind('<<ComboboxSelected>>', self.cambiar_nota)
        
        self.btn_calibrar = tk.Button(frame_seleccion, text="🎯 CALIBRAR", 
                                     font=("Arial", 10, "bold"), command=self.iniciar_calibracion,
                                     bg="#ff9800", fg="white")
        self.btn_calibrar.pack(side=tk.LEFT, padx=10)
        
        self.lbl_freq_ideal = tk.Label(frame_seleccion, 
                                      text=f"Objetivo: {self.frecuencia_ideal} Hz", 
                                      font=("Arial", 10, "bold"), bg="#e8f4fd", fg="#d35400")
        self.lbl_freq_ideal.pack(side=tk.LEFT, padx=10)
        
        self.lbl_calibracion = tk.Label(frame_seleccion, text="", 
                                       font=("Arial", 9), bg="#e8f4fd")
        self.lbl_calibracion.pack(side=tk.LEFT, padx=5)

        # 2. Panel de Acordes
        frame_acordes = tk.Frame(self.root, bg="#fff0f5", pady=10, relief=tk.RAISED, bd=1)
        frame_acordes.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(frame_acordes, text="🎶 DETECTOR DE ACORDES:", 
                font=("Arial", 12, "bold"), bg="#fff0f5").pack()
        
        self.lbl_acorde = tk.Label(frame_acordes, text="ACORDE: ---", 
                                  font=("Arial", 24, "bold"), bg="#fff0f5", fg="#8e44ad")
        self.lbl_acorde.pack(pady=5)
        
        self.lbl_confianza = tk.Label(frame_acordes, text="Confianza: 0%", 
                                     font=("Arial", 12), bg="#fff0f5", fg="#666")
        self.lbl_confianza.pack()
        
        self.lbl_notas_detectadas = tk.Label(frame_acordes, text="Notas: --", 
                                           font=("Arial", 11), bg="#fff0f5", fg="#444")
        self.lbl_notas_detectadas.pack()
        
        self.btn_modo_acordes = tk.Button(frame_acordes, text="🎵 MODO ACORDES", 
                                         font=("Arial", 10, "bold"), command=self.toggle_modo_acordes,
                                         bg="#9b59b6", fg="white")
        self.btn_modo_acordes.pack(pady=5)

        # 3. Panel de Grabación (NUEVO)
        frame_grabacion = tk.Frame(self.root, bg="#f0fff0", pady=10, relief=tk.RAISED, bd=1)
        frame_grabacion.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(frame_grabacion, text="🎙️ GRABADORA DE AUDIO:", 
                font=("Arial", 12, "bold"), bg="#f0fff0").pack()
        
        # Controles de grabación
        frame_controles_grabacion = tk.Frame(frame_grabacion, bg="#f0fff0")
        frame_controles_grabacion.pack(pady=5)
        
        self.btn_grabar = tk.Button(frame_controles_grabacion, text="⏺️ INICIAR GRABACIÓN", 
                                   font=("Arial", 11, "bold"), command=self.toggle_grabacion,
                                   bg="#e74c3c", fg="white", width=15)
        self.btn_grabar.pack(side=tk.LEFT, padx=5)
        
        self.btn_reproducir = tk.Button(frame_controles_grabacion, text="▶️ REPRODUCIR", 
                                       font=("Arial", 11, "bold"), command=self.reproducir_grabacion,
                                       bg="#3498db", fg="white", width=12)
        self.btn_reproducir.pack(side=tk.LEFT, padx=5)
        
        self.btn_guardar = tk.Button(frame_controles_grabacion, text="💾 GUARDAR", 
                                    font=("Arial", 11, "bold"), command=self.guardar_grabacion,
                                    bg="#27ae60", fg="white", width=10)
        self.btn_guardar.pack(side=tk.LEFT, padx=5)
        
        self.btn_limpiar = tk.Button(frame_controles_grabacion, text="🗑️ LIMPIAR", 
                                    font=("Arial", 11, "bold"), command=self.limpiar_grabacion,
                                    bg="#95a5a6", fg="white", width=10)
        self.btn_limpiar.pack(side=tk.LEFT, padx=5)
        
        # Información de grabación
        frame_info_grabacion = tk.Frame(frame_grabacion, bg="#f0fff0")
        frame_info_grabacion.pack(pady=5)
        
        self.lbl_estado_grabacion = tk.Label(frame_info_grabacion, text="Listo para grabar", 
                                           font=("Arial", 11, "bold"), bg="#f0fff0", fg="#2c3e50")
        self.lbl_estado_grabacion.pack(side=tk.LEFT, padx=10)
        
        self.lbl_tiempo_grabacion = tk.Label(frame_info_grabacion, text="00:00 / 00:30", 
                                           font=("Arial", 11), bg="#f0fff0", fg="#666")
        self.lbl_tiempo_grabacion.pack(side=tk.LEFT, padx=10)
        
        self.lbl_tamano_grabacion = tk.Label(frame_info_grabacion, text="0.0 MB", 
                                           font=("Arial", 11), bg="#f0fff0", fg="#666")
        self.lbl_tamano_grabacion.pack(side=tk.LEFT, padx=10)
        
        # Configuración de duración
        frame_duracion = tk.Frame(frame_grabacion, bg="#f0fff0")
        frame_duracion.pack(pady=5)
        
        tk.Label(frame_duracion, text="Duración (segundos):", 
                font=("Arial", 9), bg="#f0fff0").pack(side=tk.LEFT)
        
        self.entry_duracion = tk.Entry(frame_duracion, width=5, font=("Arial", 9))
        self.entry_duracion.insert(0, "30")
        self.entry_duracion.pack(side=tk.LEFT, padx=5)

        # 4. Panel Superior de Información
        frame_info = tk.Frame(self.root, pady=15)
        frame_info.pack(fill=tk.X)
        
        self.lbl_nota = tk.Label(frame_info, text="--", font=("Arial", 60, "bold"), fg="#333")
        self.lbl_nota.pack()
        
        self.lbl_freq = tk.Label(frame_info, text="0.00 Hz", font=("Arial", 14), fg="#666")
        self.lbl_freq.pack()
        
        self.lbl_instruccion = tk.Label(frame_info, text="Selecciona una cuerda y presiona INICIAR", 
                                       font=("Arial", 16, "bold"), fg="#007acc")
        self.lbl_instruccion.pack(pady=5)
        
        self.lbl_precision = tk.Label(frame_info, text="", font=("Arial", 12))
        self.lbl_precision.pack()

        # 5. Barra de Afinación
        self.canvas_afinador = tk.Canvas(self.root, width=600, height=80, bg="#2b2b2b")
        self.canvas_afinador.pack(pady=15)
        
        self.canvas_afinador.create_line(300, 0, 300, 80, width=3, fill="white")
        self.canvas_afinador.create_text(300, 75, text="▼", fill="white", font=("Arial", 12))
        
        for pos in [150, 225, 375, 450]:
            self.canvas_afinador.create_line(pos, 20, pos, 60, width=1, fill="gray")
        
        self.canvas_afinador.create_text(150, 65, text="-3", fill="white", font=("Arial", 8))
        self.canvas_afinador.create_text(225, 65, text="-1.5", fill="white", font=("Arial", 8))
        self.canvas_afinador.create_text(375, 65, text="+1.5", fill="white", font=("Arial", 8))
        self.canvas_afinador.create_text(450, 65, text="+3", fill="white", font=("Arial", 8))
        
        self.aguja = self.canvas_afinador.create_polygon(300, 15, 290, 65, 310, 65, fill="white")

        # 6. Gráfica
        self.frame_grafica = tk.Frame(self.root)
        self.frame_grafica.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.ax1.set_title("Espectro de Frecuencias")
        self.ax1.set_ylabel("Magnitud")
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlim(60, 1000)
        self.ax1.grid(True, alpha=0.3)
        
        self.linea_ideal = self.ax1.axvline(x=self.frecuencia_ideal, color='green', 
                                          linestyle='--', alpha=0.7, label='Objetivo')
        
        self.x_data = np.linspace(0, FRECUENCIA_MUESTREO/2, TAMANO_VENTANA//2 + 1)
        self.line, = self.ax1.plot([], [], color='#e91e63', lw=1.5, label='Espectro')
        self.ax1.legend()
        
        self.ax2.set_title("Historial de Notas para Acordes")
        self.ax2.set_xlabel("Tiempo (muestras)")
        self.ax2.set_ylabel("Nota")
        self.ax2.set_ylim(0, 12)
        self.ax2.set_yticks(range(12))
        self.ax2.set_yticklabels(NOTAS)
        self.ax2.grid(True, alpha=0.3)
        
        self.scatter_notas = self.ax2.scatter([], [], c='blue', alpha=0.6, s=50)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_grafica)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 7. Controles Inferiores
        frame_controles = tk.Frame(self.root, bg="#ddd", pady=15, bd=1, relief=tk.RAISED)
        frame_controles.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.btn_accion = tk.Button(frame_controles, text="🎙️ INICIAR CAPTURA", font=("Arial", 11, "bold"), 
                                    command=self.toggle_audio, bg="#4CAF50", fg="white", width=15)
        self.btn_accion.pack(side=tk.LEFT, padx=20)

        frame_slider = tk.Frame(frame_controles, bg="#ddd")
        frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        tk.Label(frame_slider, text="🛡️ Sensibilidad (Gate)", bg="#ddd", font=("Arial", 9)).pack(anchor="w")
        self.slider_gate = tk.Scale(frame_slider, from_=0, to=60, orient=tk.HORIZONTAL, 
                                    bg="#ddd", highlightthickness=0, command=self.actualizar_gate_var)
        self.slider_gate.set(15)
        self.slider_gate.pack(fill=tk.X)

        self.chk_grafica = tk.Checkbutton(frame_controles, text="Ver Gráfica", 
                                          variable=self.mostrar_grafica, command=self.toggle_grafica, bg="#ddd")
        self.chk_grafica.pack(side=tk.RIGHT, padx=10)

    # --- FUNCIONES DE GRABACIÓN (NUEVO) ---
    def toggle_grabacion(self):
        """Inicia o detiene la grabación"""
        if not self.grabando:
            # Iniciar grabación
            try:
                self.duracion_grabacion = int(self.entry_duracion.get())
                if self.duracion_grabacion <= 0:
                    self.duracion_grabacion = 30
            except:
                self.duracion_grabacion = 30
            
            self.grabando = True
            self.audio_grabado = []
            self.tiempo_grabacion = 0
            
            self.btn_grabar.config(text="⏹️ DETENER GRABACIÓN", bg="#c0392b")
            self.lbl_estado_grabacion.config(text="🔴 GRABANDO...", fg="#c0392b")
            self.btn_reproducir.config(state="disabled")
            self.btn_guardar.config(state="disabled")
            
            print("Iniciando grabación...")
            
        else:
            # Detener grabación
            self.grabando = False
            self.btn_grabar.config(text="⏺️ INICIAR GRABACIÓN", bg="#e74c3c")
            self.lbl_estado_grabacion.config(text="Grabación completada", fg="#27ae60")
            self.btn_reproducir.config(state="normal")
            self.btn_guardar.config(state="normal")
            
            # Calcular tamaño del archivo
            tamano_mb = len(self.audio_grabado) * 4 / (1024 * 1024)  # Aproximado
            self.lbl_tamano_grabacion.config(text=f"{tamano_mb:.1f} MB")
            
            print(f"Grabación detenida. {len(self.audio_grabado)} muestras capturadas.")

    def reproducir_grabacion(self):
        """Reproduce la grabación actual"""
        if not self.audio_grabado:
            self.lbl_estado_grabacion.config(text="❌ No hay grabación para reproducir", fg="#e74c3c")
            return
        
        try:
            audio_array = np.array(self.audio_grabado, dtype=np.float32)
            sd.play(audio_array, samplerate=FRECUENCIA_MUESTREO)
            self.lbl_estado_grabacion.config(text="▶️ Reproduciendo...", fg="#3498db")
            
            # Programar reset del estado después de la reproducción
            duracion = len(audio_array) / FRECUENCIA_MUESTREO
            self.root.after(int(duracion * 1000), 
                          lambda: self.lbl_estado_grabacion.config(text="Reproducción completada", fg="#27ae60"))
            
        except Exception as e:
            self.lbl_estado_grabacion.config(text=f"❌ Error: {str(e)}", fg="#e74c3c")

    def guardar_grabacion(self):
        """Guarda la grabación actual en un archivo WAV"""
        if not self.audio_grabado:
            self.lbl_estado_grabacion.config(text="❌ No hay grabación para guardar", fg="#e74c3c")
            return
        
        try:
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"guitarra_{timestamp}.wav"
            filepath = os.path.join(DIRECTORIO_GRABACIONES, filename)
            
            # Convertir a array numpy y normalizar
            audio_array = np.array(self.audio_grabado, dtype=np.float32)
            audio_int = np.int16(audio_array * 32767)
            
            # Guardar como WAV
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(FRECUENCIA_MUESTREO)
                wav_file.writeframes(audio_int.tobytes())
            
            self.lbl_estado_grabacion.config(text=f"✅ Guardado: {filename}", fg="#27ae60")
            print(f"Grabación guardada como: {filepath}")
            
        except Exception as e:
            self.lbl_estado_grabacion.config(text=f"❌ Error al guardar: {str(e)}", fg="#e74c3c")

    def limpiar_grabacion(self):
        """Limpia la grabación actual"""
        self.audio_grabado = []
        self.tiempo_grabacion = 0
        self.grabando = False
        
        self.btn_grabar.config(text="⏺️ INICIAR GRABACIÓN", bg="#e74c3c")
        self.lbl_estado_grabacion.config(text="Grabación limpiada", fg="#95a5a6")
        self.lbl_tiempo_grabacion.config(text="00:00 / 00:30")
        self.lbl_tamano_grabacion.config(text="0.0 MB")
        self.btn_reproducir.config(state="disabled")
        self.btn_guardar.config(state="disabled")

    def actualizar_tiempo_grabacion(self):
        """Actualiza el contador de tiempo de grabación"""
        if self.grabando:
            self.tiempo_grabacion += 0.05  # Actualizar cada 50ms
            
            # Formatear tiempo
            tiempo_actual = int(self.tiempo_grabacion)
            tiempo_max = self.duracion_grabacion
            
            mins_actual, segs_actual = divmod(tiempo_actual, 60)
            mins_max, segs_max = divmod(tiempo_max, 60)
            
            self.lbl_tiempo_grabacion.config(
                text=f"{mins_actual:02d}:{segs_actual:02d} / {mins_max:02d}:{segs_max:02d}"
            )
            
            # Verificar si se alcanzó el tiempo máximo
            if self.tiempo_grabacion >= self.duracion_grabacion:
                self.toggle_grabacion()
            
            # Programar próxima actualización
            self.root.after(50, self.actualizar_tiempo_grabacion)

    def toggle_modo_acordes(self):
        self.modo_acordes = not self.modo_acordes
        if self.modo_acordes:
            self.btn_modo_acordes.config(text="🎸 MODO AFINACIÓN", bg="#3498db")
            self.lbl_instruccion.config(text="Toca un acorde completo...", fg="#9b59b6")
            self.historico_notas.clear()
        else:
            self.btn_modo_acordes.config(text="🎵 MODO ACORDES", bg="#9b59b6")
            self.lbl_instruccion.config(text="Toca la cuerda seleccionada...", fg="#007acc")

    def cambiar_nota(self, event=None):
        nota = self.nota_seleccionada.get()
        self.frecuencia_ideal = REFERENCIA_GUITARRA[nota]
        self.lbl_freq_ideal.config(text=f"Objetivo: {self.frecuencia_ideal} Hz")
        
        if hasattr(self, 'linea_ideal'):
            self.linea_ideal.set_xdata([self.frecuencia_ideal, self.frecuencia_ideal])
            self.canvas_plot.draw_idle()

    def iniciar_calibracion(self):
        if not self.escuchando:
            self.lbl_calibracion.config(text="❌ Primero inicia la captura de audio")
            return
            
        self.calibrando = True
        self.muestras_calibracion = []
        self.contador_calibracion = 0
        self.lbl_calibracion.config(text="🔴 CALIBRANDO... Toca la cuerda", fg="red")
        self.btn_calibrar.config(state="disabled", bg="#95a5a6")

    def finalizar_calibracion(self, frecuencia):
        self.calibrando = False
        self.frecuencia_calibrada = frecuencia
        self.frecuencia_ideal = frecuencia
        
        self.lbl_calibracion.config(text=f"✅ Calibrado: {frecuencia:.2f} Hz", fg="green")
        self.lbl_freq_ideal.config(text=f"Objetivo: {frecuencia:.2f} Hz")
        self.btn_calibrar.config(state="normal", bg="#ff9800")
        
        if hasattr(self, 'linea_ideal'):
            self.linea_ideal.set_xdata([frecuencia, frecuencia])
            self.canvas_plot.draw_idle()

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
            self.btn_accion.config(text="🎙️ INICIAR CAPTURA", bg="#4CAF50")
            self.lbl_instruccion.config(text="Pausado", fg="#666")
            if self.stream:
                self.stream.stop()
                self.stream.close()
        else:
            self.escuchando = True
            self.btn_accion.config(text="🛑 DETENER CAPTURA", bg="#e74c3c")
            modo_texto = "acorde" if self.modo_acordes else "cuerda"
            self.lbl_instruccion.config(text=f"Toca la {modo_texto}...", fg="#333")
            self.contador_silencio = 0
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, 
                                         samplerate=FRECUENCIA_MUESTREO, blocksize=TAMANO_VENTANA)
            self.stream.start()

    def audio_callback(self, indata, frames, time_info, status):
        if not self.escuchando: return
        
        raw_audio = indata[:, 0]
        
        # --- GRABACIÓN (NUEVO) ---
        if self.grabando:
            # Agregar a la grabación actual (usamos solo un canal y normalizamos)
            audio_grabacion = raw_audio.flatten().astype(np.float32)
            self.audio_grabado.extend(audio_grabacion)
        
        try:
            audio_data = aplicar_filtro(raw_audio, LOW_CUT, HIGH_CUT, FRECUENCIA_MUESTREO)
        except:
            audio_data = raw_audio
        
        volumen_rms = np.linalg.norm(audio_data)
        
        if volumen_rms < self.valor_gate:
            try:
                self.data_queue.put_nowait(("SILENCIO", None, 0, []))
            except queue.Full: 
                pass
            return

        # FFT
        ventana = np.hanning(len(audio_data))
        fft_spectrum = np.fft.rfft(audio_data * ventana)
        magnitud = np.abs(fft_spectrum)
        max_val = np.max(magnitud)
        magnitud_norm = magnitud / max_val if max_val > 0 else magnitud
        
        freqs = np.fft.rfftfreq(len(audio_data), 1 / FRECUENCIA_MUESTREO)
        magnitud[freqs < 60] = 0
        
        # Detección de frecuencia fundamental
        idx_pico = np.argmax(magnitud)
        freq_detectada = freqs[idx_pico]
        
        if 0 < idx_pico < len(magnitud) - 1:
            alpha = magnitud[idx_pico - 1]
            beta = magnitud[idx_pico]
            gamma = magnitud[idx_pico + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            freq_detectada += p * (FRECUENCIA_MUESTREO / TAMANO_VENTANA)
        
        # Detección de múltiples notas para acordes
        from scipy.signal import find_peaks
        notas_detectadas = []
        peaks, properties = find_peaks(magnitud, height=UMBRAL_ARMONICOS, distance=20)
        
        for peak in peaks:
            freq = freqs[peak]
            if 60 < freq < 1000:
                nota, _, _ = self.procesar_nota(freq)
                if nota != "--" and nota not in notas_detectadas:
                    notas_detectadas.append(nota)
        
        # Calibración
        if self.calibrando and freq_detectada > 60:
            self.muestras_calibracion.append(freq_detectada)
            self.contador_calibracion += 1
            
            progreso = min(self.contador_calibracion / MUESTRAS_CALIBRACION * 100, 100)
            self.lbl_calibracion.config(text=f"🔴 Calibrando... {progreso:.0f}%")
            
            if self.contador_calibracion >= MUESTRAS_CALIBRACION:
                frecuencia_promedio = np.median(self.muestras_calibracion)
                self.finalizar_calibracion(frecuencia_promedio)
        
        try:
            self.data_queue.put_nowait((freq_detectada, magnitud_norm, volumen_rms, notas_detectadas[:4]))
        except queue.Full:
            pass

    def procesar_nota(self, freq):
        if freq == 0: 
            return "--", self.frecuencia_ideal, 0
            
        n = 12 * np.log2(freq / A4)
        n_round = int(round(n))
        freq_teorica = A4 * 2**(n_round/12)
        
        diferencia = freq - freq_teorica
        cents = 1200 * np.log2(freq / freq_teorica) if freq_teorica > 0 else 0
        
        if abs(cents) > 50:
            cuerda_cercana = min(REFERENCIA_GUITARRA.keys(), 
                               key=lambda nota: abs(REFERENCIA_GUITARRA[nota] - freq))
            freq_cuerda = REFERENCIA_GUITARRA[cuerda_cercana]
            
            if abs(freq - freq_cuerda) < 5:
                nota_nombre = cuerda_cercana.split()[0]
                octava = cuerda_cercana.split()[1]
                return f"{nota_nombre} {octava}", freq_cuerda, freq - freq_cuerda
        
        midi = n_round + 69
        nota = NOTAS[midi % 12]
        octava = (midi // 12) - 1
        
        return f"{nota} {octava}", self.frecuencia_ideal, diferencia

    def actualizar_gui(self):
        if not self.app_running: 
            return

        try:
            dato_nuevo = False
            tipo_dato = None
            freq_temp = 0
            mag_temp = None
            notas_multiples = []
            
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                dato_nuevo = True
                if item[0] == "SILENCIO":
                    tipo_dato = "SILENCIO"
                else:
                    tipo_dato = "SONIDO"
                    freq_temp, mag_temp, _, notas_multiples = item
            
            if self.escuchando and dato_nuevo:
                if tipo_dato == "SONIDO" and freq_temp > 60:
                    self.contador_silencio = 0
                    
                    nombre, ideal, dif = self.procesar_nota(freq_temp)
                    
                    if self.modo_acordes and notas_multiples:
                        self.historico_notas.extend(notas_multiples)
                        acorde, confianza = detectar_acorde(list(self.historico_notas))
                        self.acorde_actual = acorde
                        self.confianza_acorde = confianza
                        
                        self.lbl_acorde.config(text=f"ACORDE: {acorde}")
                        self.lbl_confianza.config(text=f"Confianza: {confianza*100:.0f}%")
                        self.lbl_notas_detectadas.config(text=f"Notas: {', '.join(set(self.historico_notas))}")
                    
                    self.ultimo_estado_valido = {
                        "nombre": nombre,
                        "ideal": ideal,
                        "freq": freq_temp,
                        "dif": dif
                    }
                    
                    self.dibujar_tuner(nombre, freq_temp, ideal, dif)
                    
                    if self.mostrar_grafica.get() and mag_temp is not None:
                        self.line.set_data(self.x_data[:len(mag_temp)], mag_temp)
                        
                        if self.modo_acordes and hasattr(self, 'scatter_notas'):
                            tiempos = list(range(len(self.historico_notas)))
                            notas_numeros = [nota_a_numero(nota) for nota in self.historico_notas if nota != "--"]
                            if notas_numeros:
                                self.scatter_notas.set_offsets(np.column_stack([tiempos[-len(notas_numeros):], notas_numeros]))
                                self.ax2.set_xlim(0, max(1, len(tiempos)))
                        
                        self.canvas_plot.draw_idle()

                elif tipo_dato == "SILENCIO":
                    self.contador_silencio += 1
                    
                    if self.contador_silencio < TIEMPO_MEMORIA and self.ultimo_estado_valido:
                        self.lbl_nota.config(fg="#888") 
                        self.lbl_instruccion.config(text="Manteniendo...", fg="#888")
                    else:
                        self.reset_ui()
                        if self.mostrar_grafica.get():
                            self.line.set_data([], [])
                            self.canvas_plot.draw_idle()

        except Exception as e:
            print(f"Error GUI: {e}")
        
        # Actualizar tiempo de grabación si está activa
        if self.grabando and not hasattr(self, '_grabacion_timer'):
            self._grabacion_timer = True
            self.actualizar_tiempo_grabacion()
        
        if self.app_running:
            self.root.after(50, self.actualizar_gui)

    def dibujar_tuner(self, nombre, freq, ideal, dif):
        es_cuerda_objetivo = abs(freq - ideal) < 10
        
        icono = "🎸" if es_cuerda_objetivo else "🎵"
        color_nota = "#000" if es_cuerda_objetivo else "#555"
        
        self.lbl_nota.config(text=f"{icono} {nombre} {icono}", fg=color_nota)
        self.lbl_freq.config(text=f"{freq:.2f} Hz | Objetivo: {ideal:.2f} Hz")
        
        offset = max(min(dif, 3), -3) 
        pos_x = 300 + (offset * 50)
        
        precision_cents = 1200 * np.log2(freq / ideal) if ideal > 0 else 0
        
        if abs(precision_cents) < 5:
            color = "#00cc00"
            txt = "✨ ¡PERFECTO! ✨"
            estado_precision = f"Precisión: {abs(precision_cents):.1f} cents ✅"
        elif abs(precision_cents) < 20:
            color = "#ff9800"
            txt = "APRETAR (+)" if dif < 0 else "SOLTAR (-)"
            estado_precision = f"Precisión: {abs(precision_cents):.1f} cents ⚠️"
        else:
            color = "#ff3d00"
            txt = "APRETAR (+)" if dif < 0 else "SOLTAR (-)"
            estado_precision = f"Precisión: {abs(precision_cents):.1f} cents ❌"
        
        self.lbl_precision.config(text=estado_precision, fg=color)
        
        self.canvas_afinador.coords(self.aguja, pos_x, 15, pos_x-10, 65, pos_x+10, 65)
        self.canvas_afinador.itemconfig(self.aguja, fill=color)
        
        if self.modo_acordes:
            txt = f"Acorde: {self.acorde_actual}"
        
        self.lbl_instruccion.config(text=txt, fg=color)

    def reset_ui(self):
        self.lbl_nota.config(text="--", fg="#333")
        self.lbl_freq.config(text="0.00 Hz")
        self.lbl_instruccion.config(text="Toca la cuerda...", fg="gray")
        self.lbl_precision.config(text="")
        self.canvas_afinador.itemconfig(self.aguja, fill="gray")
        self.ultimo_estado_valido = None

if __name__ == "__main__":
    root = tk.Tk()
    app = AfinadorApp(root)
    root.mainloop()