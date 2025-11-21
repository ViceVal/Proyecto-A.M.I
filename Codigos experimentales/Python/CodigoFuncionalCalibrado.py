from tkinter import ttk
import tkinter as tk
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import stft, find_peaks, windows
import math
from collections import deque, Counter

# --- PARÁMETROS ---
FRECUENCIA_MUESTREO = 44100
TAMANO_BLOQUE = 4096
NPERSEG = 8192
NOVERLAP = 6144
UMBRAL_VOLUMEN = 0.001

# --- TUS FRECUENCIAS REALES ---
CUERDAS_GUITARRA = {
    "E2": (161.5, 150, 175),
    "A2": (107.7, 95, 125), 
    "D3": (145.3, 130, 155),
    "G3": (199.2, 180, 220),
    "B3": (247.6, 230, 270),
    "E4": (328.4, 300, 350)
}

# --- SISTEMA DE DETECCIÓN DE ACORDES MEJORADO ---
ACORDES_ESPERADOS = {
    # Acorde E Mayor (todas las cuerdas al aire)
    "E Mayor": {
        "cuerdas": ["E2", "A2", "D3", "G3", "B3", "E4"],
        "notas": ["E", "A", "D", "G", "B", "E"],
        "descripcion": "Todas las cuerdas al aire"
    },
    # Acorde A Mayor (forma típica)
    "A Mayor": {
        "cuerdas": ["E2", "A2", "D3", "G3", "B3", "E4"],  # Ajustaremos los patrones
        "notas": ["E", "A", "D", "G", "B", "E"],
        "descripcion": "Cuerdas 2-3-4 presionadas en traste 2"
    },
    # Acorde D Mayor 
    "D Mayor": {
        "cuerdas": ["E2", "A2", "D3", "G3", "B3", "E4"],
        "notas": ["E", "A", "D", "G", "B", "E"],
        "descripcion": "Cuerdas 1-2-3 al aire"
    },
    # Acorde G Mayor
    "G Mayor": {
        "cuerdas": ["E2", "A2", "D3", "G3", "B3", "E4"],
        "notas": ["E", "A", "D", "G", "B", "E"],
        "descripcion": "Varias posiciones"
    },
    # Acorde C Mayor
    "C Mayor": {
        "cuerdas": ["E2", "A2", "D3", "G3", "B3", "E4"],
        "notas": ["E", "A", "D", "G", "B", "E"],
        "descripcion": "Forma típica de C"
    }
}

class DetectorAcordesMejorado:
    def __init__(self):
        self.historico_acordes = deque(maxlen=5)
        self.historico_cuerdas = deque(maxlen=5)
        
    def detectar_cuerdas_activadas(self, audio_data):
        """Detecta qué cuerdas están sonando basado en tus frecuencias reales"""
        try:
            f, magnitudes = self.calcular_espectro(audio_data)
            if f is None:
                return []
            
            cuerdas_detectadas = []
            frecuencias_detectadas = []
            
            # Buscar picos en el espectro
            umbral = 0.02 * np.max(magnitudes)
            picos, propiedades = find_peaks(magnitudes, height=umbral, distance=10)
            
            if len(picos) > 0 and 'peak_heights' in propiedades:
                for i, pico in enumerate(picos):
                    if pico < len(f):
                        freq = f[pico]
                        if 90 <= freq <= 380:  # Rango de tu guitarra
                            cuerda = self.identificar_cuerda_por_frecuencia(freq)
                            if cuerda and cuerda not in cuerdas_detectadas:
                                cuerdas_detectadas.append(cuerda)
                                frecuencias_detectadas.append(freq)
            
            # Ordenar por la cuerda más grave a más aguda
            orden_cuerdas = ["E2", "A2", "D3", "G3", "B3", "E4"]
            cuerdas_detectadas.sort(key=lambda x: orden_cuerdas.index(x) if x in orden_cuerdas else 99)
            
            return cuerdas_detectadas, frecuencias_detectadas
            
        except Exception as e:
            print(f"Error detectando cuerdas: {e}")
            return [], []
    
    def identificar_cuerda_por_frecuencia(self, frecuencia):
        """Identifica la cuerda basada en tus frecuencias reales"""
        mejor_cuerda = None
        menor_error = float('inf')
        
        for cuerda, (freq_objetivo, min_freq, max_freq) in CUERDAS_GUITARRA.items():
            # Primero verificar si está en el rango
            if min_freq <= frecuencia <= max_freq:
                error = abs(frecuencia - freq_objetivo)
                if error < menor_error:
                    menor_error = error
                    mejor_cuerda = cuerda
        
        # Si no está en rangos, buscar la más cercana
        if mejor_cuerda is None:
            for cuerda, (freq_objetivo, _, _) in CUERDAS_GUITARRA.items():
                error = abs(frecuencia - freq_objetivo)
                if error < menor_error:
                    menor_error = error
                    mejor_cuerda = cuerda
        
        return mejor_cuerda
    
    def calcular_espectro(self, audio_data):
        """Calcula el espectro de frecuencia"""
        try:
            if len(audio_data) < NPERSEG:
                audio_data = np.pad(audio_data, (0, NPERSEG - len(audio_data)), mode='constant')
            
            f, t, Zxx = stft(audio_data, 
                            fs=FRECUENCIA_MUESTREO,
                            nperseg=NPERSEG,
                            noverlap=NOVERLAP,
                            window='hann')
            
            magnitudes = np.mean(np.abs(Zxx), axis=1)
            return f, magnitudes
            
        except Exception as e:
            print(f"Error en cálculo de espectro: {e}")
            return None, None
    
    def identificar_acorde(self, cuerdas_activadas, frecuencias):
        """Identifica el acorde basado en las cuerdas activadas"""
        if not cuerdas_activadas:
            return "No detectado", "Ninguna cuerda detectada"
        
        print(f"Cuerdas detectadas: {cuerdas_activadas}")
        print(f"Frecuencias: {[f'{f:.1f}Hz' for f in frecuencias]}")
        
        # Patrones de acordes comunes (basados en qué cuerdas suenan)
        patrones_acordes = {
            # E Mayor - todas las cuerdas
            "E Mayor": {"E2", "A2", "D3", "G3", "B3", "E4"},
            # A Mayor - típicamente no suena E2 y E4 muy suave
            "A Mayor": {"A2", "D3", "G3", "B3", "E4"},
            # D Mayor - típicamente E2 y A2 no suenan, E4 suave
            "D Mayor": {"D3", "G3", "B3", "E4"},
            # G Mayor - todas menos E2 muy suave
            "G Mayor": {"A2", "D3", "G3", "B3", "E4"},
            # C Mayor - típicamente E2 no suena, A2 suave
            "C Mayor": {"A2", "D3", "G3", "B3"},
            # E menor - similar a E Mayor
            "E menor": {"E2", "A2", "D3", "G3", "B3", "E4"},
            # A menor
            "A menor": {"A2", "D3", "G3", "B3", "E4"},
        }
        
        # Convertir a set para comparación
        cuerdas_set = set(cuerdas_activadas)
        
        # Buscar coincidencia exacta primero
        for acorde, patron in patrones_acordes.items():
            if cuerdas_set == patron:
                self.historico_acordes.append(acorde)
                return acorde, f"Coincidencia exacta con {acorde}"
        
        # Buscar mejor coincidencia
        mejor_acorde = None
        mejor_puntaje = 0
        
        for acorde, patron in patrones_acordes.items():
            interseccion = cuerdas_set.intersection(patron)
            puntaje = len(interseccion) / len(patron)
            
            if puntaje > mejor_puntaje and puntaje > 0.6:  # Al menos 60% de coincidencia
                mejor_puntaje = puntaje
                mejor_acorde = acorde
        
        if mejor_acorde:
            self.historico_acordes.append(mejor_acorde)
            
            # Verificar estabilidad en el histórico
            if len(self.historico_acordes) >= 3:
                conteo = Counter(self.historico_acordes)
                acorde_estable, count = conteo.most_common(1)[0]
                if count >= 2:  # Al menos 2 de 3 coinciden
                    return acorde_estable, f"Posible {acorde_estable} ({mejor_puntaje*100:.0f}% coincidencia)"
            
            return f"Posible {mejor_acorde}", f"{mejor_puntaje*100:.0f}% de coincidencia"
        
        # Si no hay coincidencia buena, mostrar las cuerdas detectadas
        return "No identificado", f"Cuerdas: {', '.join(cuerdas_activadas)}"

class ProcesadorGuitarraAcustica:
    def __init__(self):
        self.historico_fundamentales = deque(maxlen=5)
        self.historico_cuerdas = deque(maxlen=5)
        self.detector_acordes = DetectorAcordesMejorado()
        
    def calcular_espectro_seguro(self, audio_data):
        """STFT con manejo seguro de errores"""
        try:
            if len(audio_data) < NPERSEG:
                audio_data = np.pad(audio_data, (0, NPERSEG - len(audio_data)), mode='constant')
            
            f, t, Zxx = stft(audio_data, 
                            fs=FRECUENCIA_MUESTREO,
                            nperseg=NPERSEG,
                            noverlap=NOVERLAP,
                            window='hann')
            
            magnitudes = np.mean(np.abs(Zxx), axis=1)
            return f, magnitudes
            
        except Exception as e:
            return None, None
    
    def detectar_fundamental_segura(self, audio_data):
        """Detección robusta de fundamental"""
        try:
            f, magnitudes = self.calcular_espectro_seguro(audio_data)
            if f is None:
                return None, None
            
            mascara = (f >= 90) & (f <= 380)
            if not np.any(mascara):
                return None, None
                
            f_guitarra = f[mascara]
            mag_guitarra = magnitudes[mascara]
            
            altura_minima = 0.05 * np.max(mag_guitarra)
            
            if len(mag_guitarra) < 3:
                return None, None
                
            picos, propiedades = find_peaks(mag_guitarra, 
                                          height=altura_minima,
                                          distance=3)
            
            if len(picos) == 0:
                return None, None
            
            if 'peak_heights' in propiedades and len(propiedades['peak_heights']) > 0:
                idx_principal = picos[np.argmax(propiedades['peak_heights'])]
                if idx_principal < len(f_guitarra):
                    frecuencia = f_guitarra[idx_principal]
                    cuerda = self.identificar_cuerda_segura(frecuencia)
                    return frecuencia, cuerda
                    
        except Exception as e:
            print(f"Error en detección fundamental: {e}")
            
        return None, None
    
    def identificar_cuerda_segura(self, frecuencia):
        """Identificación de cuerda basada en tus frecuencias"""
        try:
            mejor_cuerda = None
            menor_distancia = float('inf')
            
            for cuerda, (freq_objetivo, min_freq, max_freq) in CUERDAS_GUITARRA.items():
                if min_freq <= frecuencia <= max_freq:
                    distancia = abs(math.log2(frecuencia / freq_objetivo))
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        mejor_cuerda = cuerda
            
            if mejor_cuerda is None:
                for cuerda, (freq_objetivo, _, _) in CUERDAS_GUITARRA.items():
                    if frecuencia > 0:
                        distancia = abs(math.log2(frecuencia / freq_objetivo))
                        if distancia < menor_distancia:
                            menor_distancia = distancia
                            mejor_cuerda = cuerda
                            
            return mejor_cuerda
        except:
            return None
    
    def procesar_estabilidad_segura(self, frecuencia, cuerda):
        """Filtrado de estabilidad"""
        try:
            if frecuencia and cuerda:
                self.historico_fundamentales.append(frecuencia)
                self.historico_cuerdas.append(cuerda)
                
                if len(self.historico_fundamentales) >= 3:
                    conteo = Counter(self.historico_cuerdas)
                    if conteo:
                        cuerda_estable, count = conteo.most_common(1)[0]
                        
                        if count >= len(self.historico_cuerdas) * 0.6:
                            frecuencias_filtradas = [
                                f for f, c in zip(self.historico_fundamentales, self.historico_cuerdas)
                                if c == cuerda_estable
                            ]
                            
                            if frecuencias_filtradas:
                                frecuencia_estable = np.median(frecuencias_filtradas)
                                return frecuencia_estable, cuerda_estable
                                
        except Exception as e:
            print(f"Error en estabilidad: {e}")
            
        return frecuencia, cuerda

    def detectar_acorde_completo(self, audio_data):
        """Detección completa de acorde usando el nuevo sistema"""
        return self.detector_acordes.detectar_acorde(audio_data)

# --- FUNCIONES AUXILIARES ---
def calcular_desviacion_cents(f_medida, f_objetivo):
    if f_medida <= 0 or f_objetivo <= 0:
        return 0
    try:
        return 1200 * math.log2(f_medida / f_objetivo)
    except:
        return 0

def obtener_frecuencia_objetivo(cuerda):
    if cuerda in CUERDAS_GUITARRA:
        return CUERDAS_GUITARRA[cuerda][0]
    return 0

class AfinadorGuitarraAcustica:
    def __init__(self, root):
        self.root = root
        self.root.title("🎸 Afinador - Detector de Acordes Mejorado")
        self.root.geometry("750x600")
        
        self.escuchando = False
        self.modo_actual = "afinador"
        self.stream = None
        
        self.procesador = ProcesadorGuitarraAcustica()
        
        self.crear_interfaz()
        self.inicializar_audio()

    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, 
                              text="🎸 DETECTOR DE ACORDES - USANDO TUS FRECUENCIAS REALES", 
                              font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # Información de frecuencias
        info_frame = tk.Frame(main_frame, bg="#f8f9fa", relief=tk.RAISED, bd=1)
        info_frame.pack(fill=tk.X, pady=5, padx=10)
        
        info_text = "Tus frecuencias: " + " | ".join([f"{c}: {f[0]}Hz" for c, f in CUERDAS_GUITARRA.items()])
        tk.Label(info_frame, text=info_text, font=("Arial", 8), bg="#f8f9fa").pack(pady=3)
        
        mode_frame = tk.Frame(main_frame)
        mode_frame.pack(pady=10)
        
        self.btn_afinador = tk.Button(mode_frame, 
                                     text="🎵 AFINAR CUERDAS", 
                                     command=self.activar_modo_afinador,
                                     bg="#3498db", fg="white", 
                                     font=("Arial", 10),
                                     width=15)
        self.btn_afinador.pack(side=tk.LEFT, padx=5)
        
        self.btn_acordes = tk.Button(mode_frame, 
                                    text="🎶 DETECTAR ACORDES", 
                                    command=self.activar_modo_acordes,
                                    bg="#95a5a6", fg="white", 
                                    font=("Arial", 10),
                                    width=15)
        self.btn_acordes.pack(side=tk.LEFT, padx=5)
        
        # Panel de detección de acordes
        self.frame_acordes = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        
        tk.Label(self.frame_acordes, 
                text="DETECCIÓN DE ACORDES", 
                font=("Arial", 12, "bold"),
                bg="#ecf0f1").pack(pady=10)
        
        self.lbl_acorde_principal = tk.Label(self.frame_acordes, 
                                           text="--", 
                                           font=("Arial", 28, "bold"),
                                           bg="#ecf0f1", fg="#2c3e50")
        self.lbl_acorde_principal.pack(pady=5)
        
        self.lbl_info_acorde = tk.Label(self.frame_acordes, 
                                      text="Toca un acorde...", 
                                      font=("Arial", 10),
                                      bg="#ecf0f1", fg="#7f8c8d")
        self.lbl_info_acorde.pack(pady=2)
        
        self.lbl_cuerdas_detectadas = tk.Label(self.frame_acordes, 
                                             text="Cuerdas: --", 
                                             font=("Arial", 9),
                                             bg="#ecf0f1", fg="#34495e")
        self.lbl_cuerdas_detectadas.pack(pady=2)
        
        self.lbl_frecuencias_detectadas = tk.Label(self.frame_acordes, 
                                                 text="Frecuencias: --", 
                                                 font=("Arial", 8),
                                                 bg="#ecf0f1", fg="#95a5a6")
        self.lbl_frecuencias_detectadas.pack(pady=2)
        
        # Display principal para modo afinador
        self.frame_afinador = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        
        self.lbl_resultado = tk.Label(self.frame_afinador, 
                                     text="--", 
                                     font=("Arial", 32, "bold"), 
                                     bg="#ecf0f1", fg="#7f8c8d")
        self.lbl_resultado.pack(pady=10)
        
        info_frame_af = tk.Frame(self.frame_afinador, bg="#ecf0f1")
        info_frame_af.pack(pady=8)
        
        self.lbl_frecuencia = tk.Label(info_frame_af, 
                                      text="0.00 Hz", 
                                      font=("Arial", 11),
                                      bg="#ecf0f1")
        self.lbl_frecuencia.pack(side=tk.LEFT, padx=10)
        
        self.lbl_cuerda = tk.Label(info_frame_af, 
                                  text="--", 
                                  font=("Arial", 11, "bold"),
                                  bg="#ecf0f1")
        self.lbl_cuerda.pack(side=tk.LEFT, padx=10)
        
        self.lbl_desviacion = tk.Label(info_frame_af, 
                                      text="±0 cents", 
                                      font=("Arial", 10),
                                      bg="#ecf0f1")
        self.lbl_desviacion.pack(side=tk.LEFT, padx=10)
        
        self.crear_indicador_afinacion(self.frame_afinador)
        
        self.lbl_estado = tk.Label(main_frame, 
                                  text="Presiona INICIAR para comenzar", 
                                  font=("Arial", 10))
        self.lbl_estado.pack(pady=8)
        
        self.btn_captura = tk.Button(main_frame, 
                                    text="🎙️ INICIAR CAPTURA", 
                                    command=self.toggle_captura,
                                    bg="#27ae60", fg="white", 
                                    font=("Arial", 11, "bold"),
                                    width=18, height=1)
        self.btn_captura.pack(pady=12)
        
        # Mostrar modo afinador por defecto
        self.frame_afinador.pack(fill=tk.X, pady=10, padx=10)

    def crear_indicador_afinacion(self, parent):
        frame = tk.Frame(parent, bg="#ecf0f1")
        frame.pack(pady=10)
        
        self.canvas_afinacion = tk.Canvas(frame, width=300, height=60, bg="#2c3e50")
        self.canvas_afinacion.pack()
        
        for cents in [-50, -25, 0, 25, 50]:
            x = 150 + cents
            color = "#e74c3c" if cents == 0 else "#7f8c8d"
            self.canvas_afinacion.create_line(x, 20, x, 40, fill=color, width=2)
            self.canvas_afinacion.create_text(x, 50, text=str(cents), fill="white", font=("Arial", 8))
        
        self.aguja = self.canvas_afinacion.create_line(150, 15, 150, 45, fill="#3498db", width=3)

    def activar_modo_afinador(self):
        self.modo_actual = "afinador"
        self.btn_afinador.config(bg="#3498db")
        self.btn_acordes.config(bg="#95a5a6")
        self.frame_acordes.pack_forget()
        self.frame_afinador.pack(fill=tk.X, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Afinador - Toca una cuerda individual")

    def activar_modo_acordes(self):
        self.modo_actual = "acordes"
        self.btn_afinador.config(bg="#95a5a6")
        self.btn_acordes.config(bg="#9b59b6")
        self.frame_afinador.pack_forget()
        self.frame_acordes.pack(fill=tk.X, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Acordes - Toca un acorde completo")

    def inicializar_audio(self):
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=FRECUENCIA_MUESTREO,
                blocksize=TAMANO_BLOQUE,
                latency='low'
            )
            print("✅ Sistema de audio inicializado")
        except Exception as e:
            print(f"Error audio: {e}")

    def audio_callback(self, indata, frames, time, status):
        if not self.escuchando:
            return
            
        try:
            audio_data = indata[:, 0].flatten()
            volumen = np.sqrt(np.mean(audio_data**2))
            
            if volumen < UMBRAL_VOLUMEN:
                self.root.after(0, self.mostrar_silencio)
                return
                
            if self.modo_actual == "afinador":
                frecuencia, cuerda = self.procesador.detectar_fundamental_segura(audio_data)
                if frecuencia and cuerda:
                    freq_estable, cuerda_estable = self.procesador.procesar_estabilidad_segura(frecuencia, cuerda)
                    if freq_estable and cuerda_estable:
                        self.root.after(0, lambda: self.mostrar_afinacion(freq_estable, cuerda_estable))
                else:
                    self.root.after(0, self.mostrar_silencio)
                    
            else:
                # Modo detección de acordes
                cuerdas, frecuencias = self.procesador.detector_acordes.detectar_cuerdas_activadas(audio_data)
                acorde, info = self.procesador.detector_acordes.identificar_acorde(cuerdas, frecuencias)
                self.root.after(0, lambda: self.mostrar_acorde(acorde, info, cuerdas, frecuencias))
                    
        except Exception as e:
            print(f"Error en callback: {e}")
            self.root.after(0, self.mostrar_silencio)

    def toggle_captura(self):
        if not self.escuchando:
            self.escuchando = True
            self.stream.start()
            self.btn_captura.config(text="🛑 DETENER", bg="#e74c3c")
            estado = "Escuchando... " + ("Toca una cuerda" if self.modo_actual == "afinador" else "Toca un acorde")
            self.lbl_estado.config(text=estado)
        else:
            self.detener_captura()

    def detener_captura(self):
        self.escuchando = False
        if self.stream:
            self.stream.stop()
        self.btn_captura.config(text="🎙️ INICIAR", bg="#27ae60")
        self.lbl_estado.config(text="Captura detenida")
        self.mostrar_silencio()

    def mostrar_silencio(self):
        if self.modo_actual == "afinador":
            self.lbl_resultado.config(text="--", fg="#7f8c8d")
            self.lbl_frecuencia.config(text="0.00 Hz")
            self.lbl_cuerda.config(text="--")
            self.lbl_desviacion.config(text="±0 cents")
            self.canvas_afinacion.coords(self.aguja, 150, 15, 150, 45)
            self.canvas_afinacion.itemconfig(self.aguja, fill="#3498db")
        else:
            self.lbl_acorde_principal.config(text="--")
            self.lbl_info_acorde.config(text="Toca un acorde...")
            self.lbl_cuerdas_detectadas.config(text="Cuerdas: --")
            self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")
        
        self.lbl_estado.config(text="Toca tu guitarra...")

    def mostrar_afinacion(self, frecuencia, cuerda):
        try:
            self.lbl_resultado.config(text=cuerda, fg="#2c3e50")
            self.lbl_frecuencia.config(text=f"{frecuencia:.1f} Hz")
            self.lbl_cuerda.config(text=cuerda)
            
            freq_objetivo = obtener_frecuencia_objetivo(cuerda)
            cents = calcular_desviacion_cents(frecuencia, freq_objetivo)
            
            desplazamiento = cents * 0.5
            nueva_x = max(50, min(250, 150 + desplazamiento))
            self.canvas_afinacion.coords(self.aguja, nueva_x, 15, nueva_x, 45)
            
            if abs(cents) < 3:
                color = "#2ecc71"
                accion = "PERFECTO"
            elif abs(cents) < 10:
                color = "#f39c12"
                direccion = "APRETAR" if cents > 0 else "AFLOJAR"
                accion = f"{direccion}"
            else:
                color = "#e74c3c"
                direccion = "APRETAR" if cents > 0 else "AFLOJAR"
                accion = f"{direccion} FUERTE"
                
            self.canvas_afinacion.itemconfig(self.aguja, fill=color)
            self.lbl_desviacion.config(text=f"{accion} ({cents:+.1f} cents)")
            
        except Exception as e:
            print(f"Error en mostrar_afinacion: {e}")
            self.mostrar_silencio()

    def mostrar_acorde(self, acorde, info, cuerdas, frecuencias):
        try:
            self.lbl_acorde_principal.config(text=acorde)
            self.lbl_info_acorde.config(text=info)
            
            if cuerdas:
                self.lbl_cuerdas_detectadas.config(text=f"Cuerdas: {', '.join(cuerdas)}")
                frec_str = ", ".join([f"{f:.1f}Hz" for f in frecuencias])
                self.lbl_frecuencias_detectadas.config(text=f"Frecuencias: {frec_str}")
            else:
                self.lbl_cuerdas_detectadas.config(text="Cuerdas: --")
                self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")
            
            self.lbl_estado.config(text=f"Acorde: {acorde}")
            
        except Exception as e:
            print(f"Error en mostrar_acorde: {e}")
            self.mostrar_silencio()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AfinadorGuitarraAcustica(root)
        
        def on_closing():
            app.detener_captura()
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
