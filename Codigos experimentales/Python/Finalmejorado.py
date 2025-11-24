from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import sounddevice as sd
import numpy as np
from scipy.signal import stft, find_peaks, butter, lfilter, fftconvolve
import math
from collections import deque, Counter
import time
from typing import Dict, List, Tuple, Optional, Deque, Any
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- PARÁMETROS ORIGINALES ---
FRECUENCIA_MUESTREO = 44100
TAMANO_BLOQUE = 4096
NPERSEG = 8192
NOVERLAP = 6144
UMBRAL_VOLUMEN = 0.001

# --- FRECUENCIAS DE NOTAS MUSICALES ---
NOTAS_MUSICALES: Dict[str, float] = {
    "C": 130.81,   "C#": 138.59,  "D": 146.83,   "D#": 155.56,
    "E": 164.81,   "F": 174.61,   "F#": 185.00,  "G": 196.00,
    "G#": 207.65,  "A": 220.00,   "A#": 233.08,  "B": 246.94,
}

# --- FRECUENCIAS REALES DE CUERDAS (VALORES EMPÍRICOS CALIBRADOS) ---
CUERDAS_GUITARRA: Dict[str, Tuple[float, int, int]] = {
    "E2": (161.5, 150, 175),   # Tu valor empírico
    "A2": (220.0, 200, 240),   # CORREGIDO: 220 Hz para A2
    "D3": (145.3, 130, 165),   # Tu valor empírico  
    "G3": (199.2, 180, 220),   # Tu valor empírico
    "B3": (247.6, 230, 270),   # Tu valor empírico
    "E4": (328.4, 300, 350)    # Tu valor empírico
}

# --- ACORDES (SISTEMA ORIGINAL QUE FUNCIONABA BIEN) ---
ACORDES_GUITARRA: Dict[str, Dict[str, Any]] = {
    "E Mayor": {"notas": ["E", "G#", "B"], "tipo": "mayor", "prioridad": 1},
    "A Mayor": {"notas": ["A", "C#", "E"], "tipo": "mayor", "prioridad": 1},
    "D Mayor": {"notas": ["D", "F#", "A"], "tipo": "mayor", "prioridad": 1},
    "G Mayor": {"notas": ["G", "B", "D"], "tipo": "mayor", "prioridad": 1},
    "C Mayor": {"notas": ["C", "E", "G"], "tipo": "mayor", "prioridad": 1},
    "F Mayor": {"notas": ["F", "A", "C"], "tipo": "mayor", "prioridad": 1},
    "B Mayor": {"notas": ["B", "D#", "F#"], "tipo": "mayor", "prioridad": 1},
    "E menor": {"notas": ["E", "G", "B"], "tipo": "menor", "prioridad": 1},
    "A menor": {"notas": ["A", "C", "E"], "tipo": "menor", "prioridad": 1},
    "D menor": {"notas": ["D", "F", "A"], "tipo": "menor", "prioridad": 1},
    "G menor": {"notas": ["G", "A#", "D"], "tipo": "menor", "prioridad": 1},
    "C menor": {"notas": ["C", "D#", "G"], "tipo": "menor", "prioridad": 1},
    "F menor": {"notas": ["F", "G#", "C"], "tipo": "menor", "prioridad": 1},
    "B menor": {"notas": ["B", "D", "F#"], "tipo": "menor", "prioridad": 1},
}

class AnalizadorFourier:
    """Implementa las Series de Fourier: f(t) = a0 + Σ [an cos(nω0t) + bn sin(nω0t)]"""
    
    def __init__(self):
        self.num_armonicos = 10
        
    def calcular_series_fourier(self, audio_data: np.ndarray, frecuencia_fundamental: float) -> Dict[str, Any]:
        """Calcula los coeficientes de Fourier para una señal"""
        try:
            N = len(audio_data)
            if N == 0:
                return {}
            
            T = N / FRECUENCIA_MUESTREO
            t = np.linspace(0, T, N, endpoint=False)
            omega_0 = 2 * math.pi * frecuencia_fundamental
            a0 = np.mean(audio_data)
            
            a_n_list: List[float] = []
            b_n_list: List[float] = []
            armonicos_list: List[int] = []
            magnitudes_list: List[float] = []
            
            coeficientes = {
                'a0': float(a0),
                'a_n': a_n_list,
                'b_n': b_n_list,
                'armonicos': armonicos_list,
                'magnitudes': magnitudes_list,
                'frecuencia_fundamental': frecuencia_fundamental
            }
            
            for n in range(1, self.num_armonicos + 1):
                audio_float = audio_data.astype(np.float64)
                t_float = t.astype(np.float64)
                n_int = int(n)
                cos_term_float = np.cos(n_int * omega_0 * t_float).astype(np.float64)
                sin_term_float = np.sin(n_int * omega_0 * t_float).astype(np.float64)
                
                an_val = (2.0 / T) * np.trapezoid(audio_float * cos_term_float, t_float)
                bn_val = (2.0 / T) * np.trapezoid(audio_float * sin_term_float, t_float)
                magnitud = math.sqrt(float(an_val)**2 + float(bn_val)**2)
                
                coeficientes['a_n'].append(float(an_val))
                coeficientes['b_n'].append(float(bn_val))
                coeficientes['armonicos'].append(n_int)
                coeficientes['magnitudes'].append(float(magnitud))
            
            return coeficientes
            
        except Exception as error:
            print(f"Error en series de Fourier: {error}")
            return {}
    
    def analizar_timbre_mejorado(self, coeficientes: Dict[str, Any]) -> Dict[str, float]:
        """Análisis de timbre con fundamentos matemáticos sólidos"""
        try:
            if not coeficientes or 'magnitudes' not in coeficientes:
                return {}
            
            magnitudes = coeficientes['magnitudes']
            if not magnitudes:
                return {}
            
            magnitudes_arr = np.array(magnitudes, dtype=np.float64)
            if len(magnitudes_arr) == 0:
                return {}
            
            # 1. Centroide espectral (medida de brillantez)
            frecuencias_armonicos = coeficientes['frecuencia_fundamental'] * np.arange(1, len(magnitudes_arr)+1)
            if np.sum(magnitudes_arr) > 0:
                centroide_espectral = np.sum(frecuencias_armonicos * magnitudes_arr) / np.sum(magnitudes_arr)
            else:
                centroide_espectral = 0.0
            
            # 2. Ancho de banda espectral
            if np.sum(magnitudes_arr) > 0:
                desviacion = np.sqrt(np.sum(magnitudes_arr * (frecuencias_armonicos - centroide_espectral)**2) / np.sum(magnitudes_arr))
            else:
                desviacion = 0.0
            
            # 3. Relación fundamental/armónicos
            magnitud_fundamental = magnitudes_arr[0] if len(magnitudes_arr) > 0 else 0.0
            magnitud_total = np.sum(magnitudes_arr)
            relacion_fundamental = magnitud_fundamental / magnitud_total if magnitud_total > 0 else 0.0
            
            # 4. Medida de inharmonicidad (importante para cuerdas reales)
            if len(magnitudes_arr) >= 2:
                frecuencias_teoricas = coeficientes['frecuencia_fundamental'] * np.arange(1, len(magnitudes_arr)+1)
                desviaciones_armonicos = np.abs(frecuencias_armonicos - frecuencias_teoricas) / frecuencias_teoricas
                if len(desviaciones_armonicos) > 3:
                    inharmonicidad = np.mean(desviaciones_armonicos[1:4])  # Primeros armónicos
                else:
                    inharmonicidad = np.mean(desviaciones_armonicos[1:]) if len(desviaciones_armonicos) > 1 else 0.0
            else:
                inharmonicidad = 0.0
            
            return {
                'centroide_espectral': float(centroide_espectral),
                'ancho_banda': float(desviacion),
                'relacion_fundamental': float(relacion_fundamental),
                'inharmonicidad': float(inharmonicidad),
                'riqueza_espectral': float(np.sum(magnitudes_arr[1:]) / magnitud_total if magnitud_total > 0 else 0),
                'armonico_principal': int(np.argmax(magnitudes_arr) + 1 if len(magnitudes_arr) > 0 else 1)
            }
            
        except Exception as error:
            print(f"Error analizando timbre mejorado: {error}")
            return {}
    
    def analizar_timbre(self, coeficientes: Dict[str, Any]) -> Dict[str, float]:
        """Método original mantenido para compatibilidad"""
        return self.analizar_timbre_mejorado(coeficientes)

class DetectorNotasAcordes:
    def __init__(self):
        self.historico_acordes: Deque[str] = deque(maxlen=5)
        self.historico_notas: Deque[str] = deque(maxlen=10)
        self.ultimo_acorde_detectado: Optional[str] = None
        self.tiempo_ultimo_acorde: float = 0.0
        self.analizador_fourier = AnalizadorFourier()

    def identificar_cuerda_por_frecuencia(self, frecuencia: float) -> Optional[str]:
        """Identifica la cuerda directamente por frecuencia empírica"""
        mejor_cuerda = None
        menor_error = float('inf')
        
        for cuerda, (freq_objetivo, min_freq, max_freq) in CUERDAS_GUITARRA.items():
            if min_freq <= frecuencia <= max_freq:
                error = abs(math.log2(frecuencia / freq_objetivo))
                if error < menor_error:
                    menor_error = error
                    mejor_cuerda = cuerda
        
        return mejor_cuerda

    def frecuencia_a_nota_corregida(self, frecuencia: float) -> Tuple[Optional[str], float]:
        """Conversión frecuencia-nota CORREGIDA para respetar valores empíricos"""
        if frecuencia <= 0:
            return None, float('inf')
        
        # PRIMERO: Verificar si corresponde a alguna cuerda empírica
        cuerda_empirica = self.identificar_cuerda_por_frecuencia(frecuencia)
        
        # Si encontramos una cuerda empírica con buen match, usarla
        if cuerda_empirica:
            nota_base = cuerda_empirica[:-1]  # Remover el número de octava (ej: "A2" -> "A")
            return nota_base, 0.01  # Error bajo porque es empírico
        
        # SEGUNDO: Método tradicional para otras frecuencias
        mejor_nota: Optional[str] = None
        menor_error: float = float('inf')
        
        for nota, freq_objetivo in NOTAS_MUSICALES.items():
            for octava in range(2, 6):
                freq_octava = freq_objetivo * (2 ** (octava - 4))
                error = abs(math.log2(frecuencia / freq_octava))
                if error < menor_error:
                    menor_error = error
                    mejor_nota = nota
        
        return mejor_nota, menor_error

    def frecuencia_a_nota_mejorada(self, frecuencia: float) -> Tuple[Optional[str], float]:
        """Conversión frecuencia-nota mejorada que prioriza valores empíricos"""
        return self.frecuencia_a_nota_corregida(frecuencia)

    def frecuencia_a_nota(self, frecuencia: float) -> Tuple[Optional[str], float]:
        """Método original mantenido para compatibilidad"""
        return self.frecuencia_a_nota_mejorada(frecuencia)

    def calcular_espectro_mejorado(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """STFT optimizado"""
        try:
            if len(audio_data) < NPERSEG:
                audio_data = np.pad(audio_data, (0, NPERSEG - len(audio_data)), mode='constant')
            
            f, _, Zxx = stft(audio_data, 
                            fs=FRECUENCIA_MUESTREO,
                            nperseg=NPERSEG,
                            noverlap=NOVERLAP,
                            window='hann',
                            nfft=NPERSEG * 2)
            
            magnitudes = np.sqrt(np.mean(np.abs(Zxx)**2, axis=1))
            return f, magnitudes
            
        except Exception as error:
            print(f"Error en cálculo de espectro: {error}")
            return np.array([]), np.array([])

    def detectar_notas_en_audio(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta notas usando ANÁLISIS COMBINADO: STFT + Series de Fourier"""
        try:
            f, magnitudes = self.calcular_espectro_mejorado(audio_data)
            
            if len(f) == 0 or len(magnitudes) == 0:
                return []
            
            notas_detectadas: List[Dict[str, Any]] = []
            umbral = 0.015 * np.max(magnitudes)
            picos, propiedades = find_peaks(magnitudes, 
                                          height=umbral, 
                                          distance=8,
                                          prominence=umbral*0.5)
            
            if len(picos) > 0 and 'peak_heights' in propiedades:
                for i, pico in enumerate(picos):
                    if pico < len(f):
                        freq = f[pico]
                        if 80 <= freq <= 450:
                            # USAR MÉTODO CORREGIDO que prioriza valores empíricos
                            nota, error = self.frecuencia_a_nota_corregida(freq)
                            
                            if nota and error < 0.08:
                                magnitud = propiedades['peak_heights'][i]
                                if magnitud > umbral * 1.2:
                                    coeficientes_fourier = self.analizador_fourier.calcular_series_fourier(audio_data, freq)
                                    analisis_timbre = self.analizador_fourier.analizar_timbre_mejorado(coeficientes_fourier)
                                    
                                    # Identificar si es una cuerda empírica
                                    cuerda_empirica = self.identificar_cuerda_por_frecuencia(freq)
                                    
                                    notas_detectadas.append({
                                        'nota': nota,
                                        'frecuencia': freq,
                                        'error': error,
                                        'magnitud': magnitud,
                                        'coeficientes_fourier': coeficientes_fourier,
                                        'timbre': analisis_timbre,
                                        'confianza': self.calcular_confianza_mejorada(analisis_timbre, error, magnitud),
                                        'cuerda_empirica': cuerda_empirica
                                    })
            
            notas_detectadas.sort(key=lambda x: x['confianza'], reverse=True)
            notas_unicas: Dict[str, Dict[str, Any]] = {}
            for nota_info in notas_detectadas:
                nota = nota_info['nota']
                if nota is not None:
                    if nota not in notas_unicas or nota_info['confianza'] > notas_unicas[nota]['confianza']:
                        notas_unicas[nota] = nota_info
            
            return list(notas_unicas.values())
            
        except Exception as error:
            print(f"Error detectando notas: {error}")
            return []

    def calcular_confianza_mejorada(self, timbre: Dict[str, float], error_frecuencia: float, magnitud: float) -> float:
        """Calcula confianza mejorada basada en análisis de Fourier completo"""
        # Base por error de frecuencia
        confianza = 1.0 - min(error_frecuencia * 10, 1.0)
        
        # Factor por magnitud (notas más fuertes = más confianza)
        confianza_magnitud = min(magnitud / 0.1, 1.0) * 0.2  # Hasta 20% de bonus
        confianza += confianza_magnitud
        
        # Mejorar confianza si el timbre es consistente con instrumento de cuerda
        if timbre:
            # Cuerdas típicamente tienen buena relación fundamental
            if timbre.get('relacion_fundamental', 0) > 0.3:
                confianza += 0.15
            
            # Baja inharmonicidad indica sonido musical puro
            if timbre.get('inharmonicidad', 1.0) < 0.1:
                confianza += 0.1
            
            # Centroide espectral en rango típico de guitarra
            centroide = timbre.get('centroide_espectral', 0)
            if 200 <= centroide <= 1200:
                confianza += 0.05
        
        return min(confianza, 1.0)

    def calcular_confianza(self, timbre: Dict[str, float], error_frecuencia: float) -> float:
        """Método original mantenido para compatibilidad"""
        return self.calcular_confianza_mejorada(timbre, error_frecuencia, 0.05)

    def identificar_acorde_por_notas(self, notas_detectadas: List[Dict[str, Any]]) -> Tuple[str, str]:
        """SISTEMA ORIGINAL MEJORADO - MÁS PRECISO PARA MAYORES/MENORES"""
        if len(notas_detectadas) < 2:
            return "No detectado", "Muy pocas notas"
        
        notas_completas: List[str] = []
        
        for nota_info in notas_detectadas:
            nota = nota_info['nota']
            if nota is not None:
                notas_completas.append(nota)
        
        if not notas_completas:
            return "No detectado", "No se detectaron notas válidas"
        
        notas_set = set(notas_completas)
        print(f"Notas detectadas: {notas_set}")
        
        mejor_acorde: Optional[str] = None
        mejor_puntaje: float = 0.0
        
        # ANÁLISIS MEJORADO PARA DISTINGUIR ENTRE MAYORES Y MENORES
        for acorde, info in ACORDES_GUITARRA.items():
            notas_acorde_set = set(info["notas"])
            
            # Coincidencias exactas (más peso)
            coincidencias_exactas = notas_set.intersection(notas_acorde_set)
            puntaje_exacto = len(coincidencias_exactas) / len(info["notas"])
            
            # Coincidencias de notas base (menos peso)
            notas_base_detectadas = set([n[0] for n in notas_completas])
            notas_base_acorde = set([n[0] for n in info["notas"]])
            coincidencias_base = notas_base_detectadas.intersection(notas_base_acorde)
            puntaje_base = len(coincidencias_base) / len(info["notas"])
            
            # Puntaje combinado
            puntaje = (puntaje_exacto * 0.8) + (puntaje_base * 0.2)
            
            # BONUS ESPECÍFICOS PARA MAYORES/MENORES
            tonica_acorde = info["notas"][0]
            tercera_acorde = info["notas"][1]
            quinta_acorde = info["notas"][2]
            
            # Bonus por tónica presente
            if tonica_acorde in notas_set:
                puntaje += 0.3
            
            # Bonus CRÍTICO: tercera presente (esto diferencia mayores de menores)
            if tercera_acorde in notas_set:
                puntaje += 0.4  # Bonus alto para la tercera
            
            # Bonus por quinta
            if quinta_acorde in notas_set:
                puntaje += 0.2
            
            # PENALIZACIONES ESPECÍFICAS PARA EVITAR CONFUSIONES
            if info["tipo"] == "mayor":
                # Penalizar si aparece la tercera menor en un acorde mayor
                tercera_menor = tercera_acorde.replace("#", "") if "#" in tercera_acorde else tercera_acorde + "b"
                if tercera_menor in notas_set and tercera_acorde not in notas_set:
                    puntaje -= 0.5
            
            elif info["tipo"] == "menor":
                # Penalizar si aparece la tercera mayor en un acorde menor
                tercera_mayor = tercera_acorde.replace("b", "") if "b" in tercera_acorde else tercera_acorde + "#"
                if tercera_mayor in notas_set and tercera_acorde not in notas_set:
                    puntaje -= 0.5
            
            print(f"  {acorde}: exactas={coincidencias_exactas} ({puntaje:.1%})")
            
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_acorde = acorde
        
        # LÓGICA DE CONFIRMACIÓN MEJORADA
        tiempo_actual = time.time()
        if mejor_acorde and mejor_puntaje > 0.65:  # Umbral más bajo para mejor sensibilidad
            
            # CONFIRMACIÓN CRÍTICA: Verificar que tenga al menos tónica y tercera
            info_acorde = ACORDES_GUITARRA[mejor_acorde]
            tonica_presente = info_acorde["notas"][0] in notas_set
            tercera_presente = info_acorde["notas"][1] in notas_set
            
            if tonica_presente and tercera_presente:
                self.ultimo_acorde_detectado = mejor_acorde
                self.tiempo_ultimo_acorde = tiempo_actual
                self.historico_acordes.append(mejor_acorde)
                
                # Usar histórico para estabilizar
                if len(self.historico_acordes) >= 3:
                    conteo = Counter(self.historico_acordes)
                    acorde_estable, count = conteo.most_common(1)[0]
                    if count >= 2:
                        return acorde_estable, f"✅ {acorde_estable} ({mejor_puntaje:.0%})"
                
                if mejor_puntaje > 0.8:
                    return mejor_acorde, f"🎵 {mejor_acorde} ({mejor_puntaje:.0%})"
                elif mejor_puntaje > 0.7:
                    return mejor_acorde, f"🎵 {mejor_acorde} ({mejor_puntaje:.0%})"
                else:
                    return f"Posible {mejor_acorde}", f"{mejor_puntaje:.0%} coincidencia"
        
        # Mantener último acorde por un tiempo corto
        elif (self.ultimo_acorde_detectado and 
              tiempo_actual - self.tiempo_ultimo_acorde < 1.5):
            return self.ultimo_acorde_detectado, f"⏹ {self.ultimo_acorde_detectado} (mantenido)"
        
        return "No identificado", f"Notas: {', '.join(notas_set)}"

    def identificar_tonica_probable(self, notas: List[str]) -> str:
        """Identifica la tónica más probable basada en las notas detectadas"""
        if not notas:
            return ""
        
        notas_base = [n[0] for n in notas]
        conteo = Counter(notas_base)
        
        tonicas_comunes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for tonica in tonicas_comunes:
            if tonica in conteo:
                return tonica
        
        return conteo.most_common(1)[0][0]

    def identificar_acorde_combinado(self, notas_detectadas: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Usa exclusivamente el método empírico calibrado"""
        return self.identificar_acorde_por_notas(notas_detectadas)

class ProcesadorGuitarraAcustica:
    def __init__(self):
        self.historico_fundamentales: Deque[float] = deque(maxlen=5)
        self.historico_cuerdas: Deque[str] = deque(maxlen=5)
        self.detector_notas = DetectorNotasAcordes()
        
    def detectar_fundamental_segura(self, audio_data: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
        """Detección robusta de fundamental usando valores empíricos"""
        try:
            f, magnitudes = self.detector_notas.calcular_espectro_mejorado(audio_data)
            
            if len(f) == 0 or len(magnitudes) == 0:
                return None, None
            
            mascara = (f >= 85) & (f <= 350)
            if not np.any(mascara):
                return None, None
                
            f_guitarra = f[mascara]
            mag_guitarra = magnitudes[mascara]
            altura_minima = 0.05 * np.max(mag_guitarra)
            
            if len(mag_guitarra) < 3:
                return None, None
                
            picos, propiedades = find_peaks(mag_guitarra, 
                                          height=altura_minima,
                                          distance=3,
                                          prominence=altura_minima*0.3)
            
            if len(picos) == 0:
                return None, None
            
            if 'peak_heights' in propiedades and len(propiedades['peak_heights']) > 0:
                idx_principal = picos[np.argmax(propiedades['peak_heights'])]
                if idx_principal < len(f_guitarra):
                    frecuencia = float(f_guitarra[idx_principal])
                    cuerda = self.identificar_cuerda_segura(frecuencia)
                    return frecuencia, cuerda
                    
        except Exception as error:
            print(f"Error en detección fundamental: {error}")
            
        return None, None
    
    def identificar_cuerda_segura(self, frecuencia: float) -> Optional[str]:
        """Identificación de cuerda basada en TUS frecuencias empíricas calibradas"""
        try:
            mejor_cuerda: Optional[str] = None
            menor_distancia: float = float('inf')
            
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
        except Exception:
            return None
    
    def procesar_estabilidad_segura(self, frecuencia: Optional[float], cuerda: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
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
                                frecuencia_estable = float(np.median(frecuencias_filtradas))
                                return frecuencia_estable, cuerda_estable
                                
        except Exception as error:
            print(f"Error en estabilidad: {error}")
            
        return frecuencia, cuerda

    def detectar_acorde_completo(self, audio_data: np.ndarray) -> Tuple[str, str]:
        """Detección completa de acorde usando sistema EMPÍRICO CALIBRADO"""
        notas_detectadas = self.detector_notas.detectar_notas_en_audio(audio_data)
        return self.detector_notas.identificar_acorde_combinado(notas_detectadas)

# --- FUNCIONES AUXILIARES ---
def calcular_desviacion_cents(f_medida: float, f_objetivo: float) -> float:
    if f_medida <= 0 or f_objetivo <= 0:
        return 0.0
    try:
        return 1200 * math.log2(f_medida / f_objetivo)
    except Exception:
        return 0.0

def obtener_frecuencia_objetivo(cuerda: str) -> float:
    if cuerda in CUERDAS_GUITARRA:
        return CUERDAS_GUITARRA[cuerda][0]
    return 0.0

class ProcesadorEfectos:
    """Procesador de efectos de audio para la pedalera"""
    
    def __init__(self):
        self.samplerate = FRECUENCIA_MUESTREO
        self.audio_original: Optional[np.ndarray] = None
        self.audio_procesado: Optional[np.ndarray] = None
        self.grabando = False
        self.grabacion_actual: Optional[np.ndarray] = None
        
    def grabar_audio(self, duracion: float = 3.0) -> np.ndarray:
        """Graba audio desde el micrófono"""
        print(f"Grabando {duracion} segundos...")
        self.grabando = True
        grabacion_samples = int(duracion * self.samplerate)
        self.grabacion_actual = sd.rec(grabacion_samples, 
                                      samplerate=self.samplerate, 
                                      channels=1)
        sd.wait()
        self.grabando = False
        if self.grabacion_actual is not None:
            self.audio_original = self.grabacion_actual.flatten()
            if np.max(np.abs(self.audio_original)) > 0:
                self.audio_original = self.audio_original / np.max(np.abs(self.audio_original))
            print("Grabación completada")
            return self.audio_original.astype(np.float64)
        return np.array([], dtype=np.float64)
    
    def aplicar_efecto(self, efecto: str, parametro: float) -> np.ndarray:
        """Aplica efecto al audio grabado"""
        if self.audio_original is None:
            raise ValueError("No hay audio grabado para procesar")
            
        audio = self.audio_original.copy()
        val = parametro / 100.0
        resultado: np.ndarray = audio

        if "Pasa-Bajos" in efecto:
            cutoff = 500 + (val * 4000)
            nyq = 0.5 * self.samplerate
            normal_cutoff = cutoff / nyq
            b, a = butter(5, normal_cutoff, btype='low', analog=False)
            resultado = lfilter(b, a, audio)

        elif "Distorsión" in efecto:
            drive = 1 + (val * 20)
            resultado = np.tanh(audio * drive)
            resultado = resultado / np.max(np.abs(resultado))

        elif "Modulación" in efecto:
            freq_mod = 5 + (val * 500)
            t = np.arange(len(audio)) / self.samplerate
            carrier = np.sin(2 * np.pi * freq_mod * t)
            resultado = (audio * 0.5) + ((audio * carrier) * 0.5)

        elif "Delay" in efecto:
            delay_sec = 0.1 + (val * 0.8)
            delay_samples = int(delay_sec * self.samplerate)
            decay = 0.5
            output = np.zeros(len(audio) + delay_samples)
            output[:len(audio)] += audio
            output[delay_samples:] += audio * decay
            resultado = output

        elif "Reverb" in efecto:
            reverb_len = int(self.samplerate * (0.5 + val * 2.0))
            t = np.linspace(0, 1, reverb_len)
            ir = np.random.randn(reverb_len) * np.exp(-5 * t)
            resultado = fftconvolve(audio, ir, mode='full')
            resultado = resultado / np.max(np.abs(resultado)) * 0.9

        self.audio_procesado = resultado.astype(np.float64)
        return self.audio_procesado
    
    def reproducir_original(self):
        """Reproduce el audio original"""
        if self.audio_original is not None:
            sd.stop()
            sd.play(self.audio_original, self.samplerate)
    
    def reproducir_procesado(self):
        """Reproduce el audio procesado"""
        if self.audio_procesado is not None:
            sd.stop()
            sd.play(self.audio_procesado, self.samplerate)
    
    def guardar_audio(self, archivo: str, audio_data: np.ndarray):
        """Guarda el audio en un archivo"""
        if audio_data is not None:
            sf.write(archivo, audio_data, self.samplerate)

class AfinadorGuitarraAcustica:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎸 Analizador de Audio - Fourier + Efectos")
        self.root.geometry("900x800")
        
        self.escuchando: bool = False
        self.modo_actual: str = "afinador"
        self.stream: Optional[sd.InputStream] = None
        self.procesando_audio: bool = False
        
        self.procesador = ProcesadorGuitarraAcustica()
        self.procesador_efectos = ProcesadorEfectos()
        
        # Variables para gráficos
        self.linea_espectro: Optional[Any] = None
        self.canvas_espectro: Optional[FigureCanvasTkAgg] = None
        self.ax_espectro: Optional[Any] = None
        self.fig_espectro: Optional[Any] = None
        
        # Variables para detección de ruido
        self.ultimo_volumen = 0.0
        self.contador_ruido = 0
        self.estado_anterior = "silencio"
        
        self.crear_interfaz()
        self.inicializar_audio()

    def crear_interfaz(self) -> None:
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título principal
        title_label = tk.Label(main_frame, 
                              text="🎸 ANALIZADOR DE AUDIO - FOURIER + EFECTOS", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Menú de navegación
        menu_frame = tk.Frame(main_frame, bg="#34495e")
        menu_frame.pack(fill=tk.X, pady=5, padx=10)
        
        self.btn_afinador = tk.Button(menu_frame, 
                                     text="🎵 AFINADOR", 
                                     command=self.activar_modo_afinador,
                                     bg="#3498db", fg="white", 
                                     font=("Arial", 10, "bold"),
                                     width=12, height=2)
        self.btn_afinador.pack(side=tk.LEFT, padx=2)
        
        self.btn_acordes = tk.Button(menu_frame, 
                                    text="🎶 DETECTOR ACORDES", 
                                    command=self.activar_modo_acordes,
                                    bg="#95a5a6", fg="white", 
                                    font=("Arial", 10, "bold"),
                                    width=12, height=2)
        self.btn_acordes.pack(side=tk.LEFT, padx=2)
        
        self.btn_espectro = tk.Button(menu_frame, 
                                     text="📈 VISUALIZAR ESPECTRO", 
                                     command=self.activar_modo_espectro,
                                     bg="#95a5a6", fg="white", 
                                     font=("Arial", 10, "bold"),
                                     width=12, height=2)
        self.btn_espectro.pack(side=tk.LEFT, padx=2)
        
        self.btn_efectos = tk.Button(menu_frame, 
                                    text="🎛️ PEDALERA", 
                                    command=self.activar_modo_efectos,
                                    bg="#95a5a6", fg="white", 
                                     font=("Arial", 10, "bold"),
                                     width=12, height=2)
        self.btn_efectos.pack(side=tk.LEFT, padx=2)
        
        # Panel de detección de acordes
        self.frame_acordes = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
        
        tk.Label(self.frame_acordes, 
                text="DETECCIÓN DE ACORDES CON ANÁLISIS FOURIER", 
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
        
        self.lbl_notas_detectadas = tk.Label(self.frame_acordes, 
                                           text="Notas: --", 
                                           font=("Arial", 9),
                                           bg="#ecf0f1", fg="#34495e")
        self.lbl_notas_detectadas.pack(pady=2)
        
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
        
        # Panel de espectro - CON DETECCIÓN INTEGRADA
        self.frame_espectro = tk.Frame(main_frame, bg="#2c3e50", relief=tk.RAISED, bd=2)
        
        tk.Label(self.frame_espectro, 
                text="📈 VISUALIZADOR DE ESPECTRO + DETECTOR DE NOTAS/ACORDES", 
                font=("Arial", 14, "bold"),
                bg="#2c3e50", fg="white").pack(pady=15)
        
        # Información de detección musical
        info_musica_frame = tk.Frame(self.frame_espectro, bg="#2c3e50")
        info_musica_frame.pack(pady=10)
        
        # Acorde detectado
        self.lbl_acorde_espectro = tk.Label(info_musica_frame,
                                          text="Acorde: --",
                                          font=("Arial", 18, "bold"),
                                          bg="#2c3e50", fg="#ff9900")
        self.lbl_acorde_espectro.pack()
        
        # Información del acorde
        self.lbl_info_acorde_espectro = tk.Label(info_musica_frame,
                                               text="Información: --",
                                               font=("Arial", 12),
                                               bg="#2c3e50", fg="#cccccc")
        self.lbl_info_acorde_espectro.pack()
        
        # Notas detectadas
        self.lbl_notas_espectro = tk.Label(info_musica_frame,
                                         text="Notas detectadas: --",
                                         font=("Arial", 11),
                                         bg="#2c3e50", fg="#00ff88")
        self.lbl_notas_espectro.pack()
        
        # Frecuencia principal y volumen
        info_frecuencia_frame = tk.Frame(self.frame_espectro, bg="#2c3e50")
        info_frecuencia_frame.pack(pady=5)
        
        self.lbl_frecuencia_principal = tk.Label(info_frecuencia_frame,
                                               text="Frecuencia Principal: -- Hz",
                                               font=("Arial", 12),
                                               bg="#2c3e50", fg="#00ccff")
        self.lbl_frecuencia_principal.pack(side=tk.LEFT, padx=10)
        
        self.lbl_volumen = tk.Label(info_frecuencia_frame,
                                  text="Volumen: --",
                                  font=("Arial", 12),
                                  bg="#2c3e50", fg="#ff6666")
        self.lbl_volumen.pack(side=tk.LEFT, padx=10)
        
        # Estado de detección
        self.lbl_estado_deteccion = tk.Label(info_frecuencia_frame,
                                           text="Estado: Silencio",
                                           font=("Arial", 10),
                                           bg="#2c3e50", fg="#ffff00")
        self.lbl_estado_deteccion.pack(side=tk.LEFT, padx=10)
        
        # Gráfico de espectro
        espectro_graf_frame = tk.Frame(self.frame_espectro, bg="#2c3e50")
        espectro_graf_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Crear figura y ejes para el espectro
        self.fig_espectro, self.ax_espectro = plt.subplots(figsize=(8, 4))
        self.fig_espectro.patch.set_facecolor('#2c3e50')
        self.ax_espectro.set_facecolor('#34495e')
        
        # Configurar ejes del espectro
        self.ax_espectro.set_title("ESPECTRO DE FRECUENCIAS - DETECCIÓN MUSICAL EN TIEMPO REAL", color='white', fontsize=12)
        self.ax_espectro.set_xlim(50, 1000)
        self.ax_espectro.set_ylim(0, 1.0)
        self.ax_espectro.set_xlabel("Frecuencia (Hz)", color='white')
        self.ax_espectro.set_ylabel("Magnitud Normalizada", color='white')
        self.ax_espectro.tick_params(colors='white')
        self.ax_espectro.grid(True, alpha=0.3, color='white')
        
        # Marcar frecuencias de las cuerdas de guitarra
        for cuerda, (freq, _, _) in CUERDAS_GUITARRA.items():
            self.ax_espectro.axvline(freq, color='#555555', linestyle='--', alpha=0.7)
            self.ax_espectro.text(freq, 0.95, cuerda, color='#aaaaaa', 
                                fontsize=8, ha='center', rotation=90)
        
        # Inicializar línea vacía
        x_empty = np.linspace(50, 1000, 100)
        y_empty = np.zeros(100)
        self.linea_espectro, = self.ax_espectro.plot(x_empty, y_empty, color='#00ff88', lw=1.5, alpha=0.8)
        
        # Marcadores para picos detectados
        self.marcadores_notas, = self.ax_espectro.plot([], [], 'ro', markersize=8, alpha=0.7, label='Notas Detectadas')
        
        self.ax_espectro.legend(facecolor='#2c3e50', edgecolor='white', labelcolor='white')
        
        self.canvas_espectro = FigureCanvasTkAgg(self.fig_espectro, master=espectro_graf_frame)
        self.canvas_espectro.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Panel de efectos de sonido
        self.frame_efectos = tk.Frame(main_frame, bg="#2c3e50", relief=tk.RAISED, bd=2)
        
        tk.Label(self.frame_efectos, 
                text="🎛️ PEDALERA DE EFECTOS DSP", 
                font=("Arial", 14, "bold"),
                bg="#2c3e50", fg="white").pack(pady=15)
        
        # Controles de grabación
        grabacion_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        grabacion_frame.pack(pady=10, fill=tk.X)
        
        self.btn_grabar = tk.Button(grabacion_frame, 
                                   text="⏺️ GRABAR 3 SEGUNDOS", 
                                   command=self.grabar_audio,
                                   bg="#e74c3c", fg="white", 
                                   font=("Arial", 11, "bold"),
                                   width=20)
        self.btn_grabar.pack(side=tk.LEFT, padx=10)
        
        self.lbl_estado_grabacion = tk.Label(grabacion_frame, 
                                           text="No grabado", 
                                           font=("Arial", 10),
                                           bg="#2c3e50", fg="#ecf0f1")
        self.lbl_estado_grabacion.pack(side=tk.LEFT, padx=10)
        
        # Selector de efectos
        efectos_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        efectos_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(efectos_frame, text="Efecto:", bg="#2c3e50", fg="white", 
                font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        self.efecto_var = tk.StringVar()
        opciones = ["Filtro Pasa-Bajos", "Distorsión", "Modulación", "Delay", "Reverb"]
        self.combo_efectos = ttk.Combobox(efectos_frame, textvariable=self.efecto_var, 
                                         values=opciones, state="readonly", width=20)
        self.combo_efectos.current(0)
        self.combo_efectos.pack(side=tk.LEFT, padx=10)
        
        tk.Label(efectos_frame, text="Intensidad:", bg="#2c3e50", fg="white",
                font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        self.slider_intensidad = tk.Scale(efectos_frame, from_=0, to=100, 
                                         orient=tk.HORIZONTAL, length=150, 
                                         bg="#34495e", fg="white", 
                                         highlightbackground="#2c3e50")
        self.slider_intensidad.set(50)
        self.slider_intensidad.pack(side=tk.LEFT, padx=10)
        
        self.btn_aplicar_efecto = tk.Button(efectos_frame, 
                                          text="⚡ APLICAR EFECTO", 
                                          command=self.aplicar_efecto,
                                          bg="#9b59b6", fg="white",
                                          font=("Arial", 10, "bold"))
        self.btn_aplicar_efecto.pack(side=tk.LEFT, padx=10)
        
        # Controles de reproducción
        reproduccion_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        reproduccion_frame.pack(pady=15, fill=tk.X)
        
        self.btn_repro_original = tk.Button(reproduccion_frame, 
                                          text="▶ REPRODUCIR ORIGINAL", 
                                          command=self.reproducir_original,
                                          bg="#27ae60", fg="white",
                                          font=("Arial", 10),
                                          width=18)
        self.btn_repro_original.pack(side=tk.LEFT, padx=10)
        
        self.btn_repro_efecto = tk.Button(reproduccion_frame, 
                                        text="▶ REPRODUCIR CON EFECTO", 
                                        command=self.reproducir_con_efecto,
                                        bg="#e67e22", fg="white",
                                        font=("Arial", 10),
                                        width=20)
        self.btn_repro_efecto.pack(side=tk.LEFT, padx=10)
        
        self.btn_guardar = tk.Button(reproduccion_frame, 
                                   text="💾 GUARDAR RESULTADO", 
                                   command=self.guardar_resultado,
                                   bg="#3498db", fg="white",
                                   font=("Arial", 10))
        self.btn_guardar.pack(side=tk.LEFT, padx=10)
        
        self.btn_detener = tk.Button(reproduccion_frame, 
                                   text="⏹ DETENER", 
                                   command=sd.stop,
                                   bg="#e74c3c", fg="white",
                                   font=("Arial", 10))
        self.btn_detener.pack(side=tk.LEFT, padx=10)
        
        # Gráficas para efectos
        graficas_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        graficas_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax_original, self.ax_efecto) = plt.subplots(2, 1, figsize=(8, 4))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Configurar ejes
        for ax in [self.ax_original, self.ax_efecto]:
            ax.set_facecolor('#34495e')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.grid(True, alpha=0.3, color='white')
        
        self.ax_original.set_title("Audio Original", color='white')
        self.ax_original.set_ylim(-1.1, 1.1)
        
        self.ax_efecto.set_title("Audio con Efecto", color='white')
        self.ax_efecto.set_ylim(-1.1, 1.1)
        
        self.canvas_efectos = FigureCanvasTkAgg(self.fig, master=graficas_frame)
        self.canvas_efectos.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Estado y controles generales
        self.lbl_estado = tk.Label(main_frame, 
                                  text="Selecciona un modo y presiona INICIAR", 
                                  font=("Arial", 10))
        self.lbl_estado.pack(pady=8)
        
        controles_frame = tk.Frame(main_frame)
        controles_frame.pack(pady=10)
        
        self.btn_captura = tk.Button(controles_frame, 
                                    text="🎙️ INICIAR CAPTURA", 
                                    command=self.toggle_captura,
                                    bg="#27ae60", fg="white", 
                                    font=("Arial", 11, "bold"),
                                    width=15, height=1)
        self.btn_captura.pack(side=tk.LEFT, padx=5)
        
        self.btn_detener_main = tk.Button(controles_frame, 
                                        text="🛑 DETENER", 
                                        command=self.detener_captura,
                                        bg="#e74c3c", fg="white", 
                                        font=("Arial", 11, "bold"),
                                        width=12, height=1)
        self.btn_detener_main.pack(side=tk.LEFT, padx=5)
        
        # Mostrar modo afinador por defecto
        self.activar_modo_afinador()

    def crear_indicador_afinacion(self, parent: tk.Frame) -> None:
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

    def activar_modo_afinador(self) -> None:
        self.modo_actual = "afinador"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_afinador.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Afinador - Toca una cuerda individual de guitarra")

    def activar_modo_acordes(self) -> None:
        self.modo_actual = "acordes"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_acordes.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Acordes - Toca un acorde completo de guitarra")

    def activar_modo_espectro(self) -> None:
        self.modo_actual = "espectro"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_espectro.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Espectro - Visualizando espectro con detección musical en tiempo real")

    def activar_modo_efectos(self) -> None:
        self.modo_actual = "efectos"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_efectos.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Efectos - Graba audio y aplica efectos DSP")

    def actualizar_botones_menu(self) -> None:
        """Actualiza los colores de los botones del menú"""
        botones = {
            "afinador": self.btn_afinador,
            "acordes": self.btn_acordes,
            "espectro": self.btn_espectro,
            "efectos": self.btn_efectos
        }
        
        for modo, boton in botones.items():
            if modo == self.modo_actual:
                boton.config(bg="#3498db")
            else:
                boton.config(bg="#95a5a6")

    def ocultar_todos_frames(self) -> None:
        """Oculta todos los frames de contenido"""
        frames = [self.frame_afinador, self.frame_acordes, self.frame_espectro, self.frame_efectos]
        for frame in frames:
            frame.pack_forget()

    def inicializar_audio(self) -> None:
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=FRECUENCIA_MUESTREO,
                blocksize=TAMANO_BLOQUE,
                latency='low'
            )
            print("✅ Sistema de audio inicializado")
        except Exception as error:
            print(f"Error audio: {error}")

    def es_ruido_o_musica(self, audio_data: np.ndarray, f: np.ndarray, magnitudes: np.ndarray) -> str:
        """Determina si la señal es ruido o música basado en características espectrales"""
        try:
            volumen = np.sqrt(np.mean(audio_data**2))
            
            if volumen < UMBRAL_VOLUMEN * 2:
                return "silencio"
            
            mascara_musica = (f >= 80) & (f <= 450)
            if not np.any(mascara_musica):
                return "ruido"
            
            energia_musica = np.sum(magnitudes[mascara_musica])
            energia_total = np.sum(magnitudes)
            
            if energia_total == 0:
                return "ruido"
            
            relacion_musica = energia_musica / energia_total
            
            picos, _ = find_peaks(magnitudes[mascara_musica], 
                                height=0.1*np.max(magnitudes[mascara_musica]),
                                distance=5,
                                prominence=0.05*np.max(magnitudes[mascara_musica]))
            
            num_picos_prominentes = len(picos)
            
            if relacion_musica > 0.3 and num_picos_prominentes >= 1:
                return "musica"
            elif relacion_musica > 0.6 and num_picos_prominentes >= 2:
                return "musica"
            elif volumen > UMBRAL_VOLUMEN * 10 and num_picos_prominentes == 0:
                return "ruido"
            else:
                return "ruido"
                
        except Exception:
            return "ruido"

    def audio_callback(self, indata: np.ndarray, frames: int, callback_time: Any, status: Any) -> None:
        if not self.escuchando or self.procesando_audio:
            return
            
        self.procesando_audio = True
        try:
            audio_data = indata[:, 0]
            if audio_data is not None:
                audio_flat = audio_data.flatten()
            else:
                audio_flat = np.array([])
                
            volumen = np.sqrt(np.mean(audio_flat**2)) if len(audio_flat) > 0 else 0
            
            # VISUALIZADOR DE ESPECTRO CON DETECCIÓN MUSICAL MEJORADA
            if self.modo_actual == "espectro" and len(audio_flat) > 0:
                try:
                    f, magnitudes = self.procesador.detector_notas.calcular_espectro_mejorado(audio_flat)
                    if len(f) > 0 and len(magnitudes) > 0 and self.linea_espectro is not None:
                        
                        tipo_senal = self.es_ruido_o_musica(audio_flat, f, magnitudes)
                        
                        self.safe_after(lambda: self.lbl_estado_deteccion.config(
                            text=f"Estado: {tipo_senal.upper()}",
                            fg="#ffff00" if tipo_senal == "ruido" else "#00ff00" if tipo_senal == "musica" else "#cccccc"
                        ))
                        
                        if tipo_senal == "musica":
                            notas_detectadas = self.procesador.detector_notas.detectar_notas_en_audio(audio_flat)
                            acorde, info_acorde = self.procesador.detectar_acorde_completo(audio_flat)
                            
                            self.safe_after(lambda: self.actualizar_info_musical(acorde, info_acorde, notas_detectadas))
                            
                            mascara = (f >= 50) & (f <= 1000)
                            if np.any(mascara):
                                idx_principal = np.argmax(magnitudes[mascara])
                                freq_principal = f[mascara][idx_principal]
                                
                                self.safe_after(lambda: self.lbl_frecuencia_principal.config(
                                    text=f"Frecuencia Principal: {freq_principal:.1f} Hz"
                                ))
                                
                        elif tipo_senal == "ruido":
                            mascara = (f >= 50) & (f <= 1000)
                            if np.any(mascara):
                                idx_principal = np.argmax(magnitudes[mascara])
                                freq_principal = f[mascara][idx_principal]
                                
                                self.safe_after(lambda: self.lbl_frecuencia_principal.config(
                                    text=f"Ruido - Frecuencia dominante: {freq_principal:.1f} Hz"
                                ))
                            
                            self.safe_after(self.actualizar_info_ruido)
                            
                        else:
                            self.safe_after(self.actualizar_info_silencio)
                        
                        if np.max(magnitudes) > 0:
                            magnitudes_norm = magnitudes / np.max(magnitudes)
                        else:
                            magnitudes_norm = magnitudes
                        
                        self.linea_espectro.set_data(f, magnitudes_norm)
                        
                        if tipo_senal == "musica" and notas_detectadas:
                            frecuencias_notas = [n['frecuencia'] for n in notas_detectadas if n['nota'] is not None]
                            magnitudes_notas = [n['magnitud'] for n in notas_detectadas if n['nota'] is not None]
                            
                            if frecuencias_notas and magnitudes_notas:
                                max_mag = max(magnitudes_notas) if magnitudes_notas else 1.0
                                magnitudes_notas_norm = [mag/max_mag * 0.8 for mag in magnitudes_notas]
                                self.marcadores_notas.set_data(frecuencias_notas, magnitudes_notas_norm)
                        else:
                            self.marcadores_notas.set_data([], [])
                        
                        self.safe_after(lambda: self.lbl_volumen.config(
                            text=f"Volumen: {volumen:.4f}"
                        ))
                        
                        if self.canvas_espectro is not None:
                            self.canvas_espectro.draw_idle()
                            
                except Exception as spec_error:
                    print(f"Error en visualizador espectro: {spec_error}")
            
            if volumen < UMBRAL_VOLUMEN:
                self.safe_after(self.mostrar_silencio)
                return
                
            if self.modo_actual == "afinador" and len(audio_flat) > 0:
                frecuencia, cuerda = self.procesador.detectar_fundamental_segura(audio_flat)
                if frecuencia and cuerda:
                    freq_estable, cuerda_estable = self.procesador.procesar_estabilidad_segura(frecuencia, cuerda)
                    if freq_estable and cuerda_estable:
                        self.safe_after(lambda: self.mostrar_afinacion(freq_estable, cuerda_estable))
                else:
                    self.safe_after(self.mostrar_silencio)
                    
            elif self.modo_actual == "acordes" and len(audio_flat) > 0:
                acorde, info = self.procesador.detectar_acorde_completo(audio_flat)
                notas_detectadas = self.procesador.detector_notas.detectar_notas_en_audio(audio_flat)
                notas_nombres = [n['nota'] for n in notas_detectadas if n['nota'] is not None]
                frecuencias = [n['frecuencia'] for n in notas_detectadas]
                
                self.safe_after(lambda: self.mostrar_acorde(acorde, info, notas_nombres, frecuencias))
                    
        except Exception as error:
            print(f"Error en callback: {error}")
            self.safe_after(self.mostrar_silencio)
        finally:
            self.procesando_audio = False

    def actualizar_info_musical(self, acorde: str, info_acorde: str, notas_detectadas: List[Dict[str, Any]]) -> None:
        """Actualiza la información musical en el modo espectro"""
        try:
            self.lbl_acorde_espectro.config(text=f"Acorde: {acorde}", fg="#ff9900")
            self.lbl_info_acorde_espectro.config(text=f"Información: {info_acorde}", fg="#00ff00")
            
            if notas_detectadas:
                notas_nombres = [n['nota'] for n in notas_detectadas if n['nota'] is not None]
                if notas_nombres:
                    notas_texto = ", ".join(notas_nombres)
                    self.lbl_notas_espectro.config(text=f"Notas detectadas: {notas_texto}", fg="#00ff88")
                else:
                    self.lbl_notas_espectro.config(text="Notas detectadas: --", fg="#00ff88")
            else:
                self.lbl_notas_espectro.config(text="Notas detectadas: --", fg="#00ff88")
                
        except Exception as error:
            print(f"Error actualizando info musical: {error}")

    def actualizar_info_ruido(self) -> None:
        """Actualiza la información cuando se detecta ruido"""
        try:
            self.lbl_acorde_espectro.config(text="Acorde: RUIDO DETECTADO", fg="#ff6666")
            self.lbl_info_acorde_espectro.config(text="Información: Señal no musical", fg="#ff6666")
            self.lbl_notas_espectro.config(text="Notas detectadas: --", fg="#cccccc")
        except Exception as error:
            print(f"Error actualizando info ruido: {error}")

    def actualizar_info_silencio(self) -> None:
        """Actualiza la información cuando hay silencio"""
        try:
            self.lbl_acorde_espectro.config(text="Acorde: --", fg="#cccccc")
            self.lbl_info_acorde_espectro.config(text="Información: Silencio", fg="#cccccc")
            self.lbl_notas_espectro.config(text="Notas detectadas: --", fg="#cccccc")
            self.lbl_frecuencia_principal.config(text="Frecuencia Principal: -- Hz")
            self.lbl_volumen.config(text="Volumen: --")
        except Exception as error:
            print(f"Error actualizando info silencio: {error}")

    def safe_after(self, func) -> None:
        """Método seguro para llamar after() desde el callback"""
        try:
            if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                self.root.after(0, func)
        except Exception as error:
            print(f"Error en safe_after: {error}")

    def toggle_captura(self) -> None:
        if not self.escuchando:
            self.escuchando = True
            try:
                if self.stream:
                    self.stream.start()
                self.btn_captura.config(text="🎙️ CAPTURANDO...", bg="#e74c3c")
                estado = "Escuchando... " + {
                    "afinador": "Toca una cuerda",
                    "acordes": "Toca un acorde", 
                    "espectro": "Analizando espectro y música",
                    "efectos": "Modo pedalera"
                }[self.modo_actual]
                self.lbl_estado.config(text=estado)
            except Exception as error:
                print(f"Error al iniciar stream: {error}")
                self.escuchando = False
                self.btn_captura.config(text="🎙️ INICIAR", bg="#27ae60")
                self.lbl_estado.config(text="Error al iniciar audio")
        else:
            self.detener_captura()

    def detener_captura(self) -> None:
        """Detiene la captura de audio de forma segura"""
        self.escuchando = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception as error:
                print(f"Error al detener stream: {error}")
        self.btn_captura.config(text="🎙️ INICIAR", bg="#27ae60")
        self.lbl_estado.config(text="Captura detenida")
        self.mostrar_silencio()

    def mostrar_silencio(self) -> None:
        try:
            if self.modo_actual == "afinador":
                self.lbl_resultado.config(text="--", fg="#7f8c8d")
                self.lbl_frecuencia.config(text="0.00 Hz")
                self.lbl_cuerda.config(text="--")
                self.lbl_desviacion.config(text="±0 cents")
                self.canvas_afinacion.coords(self.aguja, 150, 15, 150, 45)
                self.canvas_afinacion.itemconfig(self.aguja, fill="#3498db")
            elif self.modo_actual == "acordes":
                tiempo_actual = time.time()
                ultimo_tiempo = self.procesador.detector_notas.tiempo_ultimo_acorde
                
                if (not self.procesador.detector_notas.ultimo_acorde_detectado or 
                    tiempo_actual - ultimo_tiempo >= 2.0):
                    self.lbl_acorde_principal.config(text="--")
                    self.lbl_info_acorde.config(text="Toca un acorde...")
                    self.lbl_notas_detectadas.config(text="Notas: --")
                    self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")
            elif self.modo_actual == "espectro":
                self.actualizar_info_silencio()
                self.lbl_estado_deteccion.config(text="Estado: Silencio", fg="#cccccc")
                if self.linea_espectro is not None:
                    x_empty = np.linspace(50, 1000, 100)
                    y_empty = np.zeros(100)
                    self.linea_espectro.set_data(x_empty, y_empty)
                    self.marcadores_notas.set_data([], [])
                    if self.canvas_espectro is not None:
                        self.canvas_espectro.draw_idle()
            
            self.lbl_estado.config(text="Toca tu guitarra o habla al micrófono...")
        except Exception as error:
            print(f"Error en mostrar_silencio: {error}")

    def mostrar_afinacion(self, frecuencia: float, cuerda: str) -> None:
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
            
        except Exception as error:
            print(f"Error en mostrar_afinacion: {error}")
            self.mostrar_silencio()

    def mostrar_acorde(self, acorde: str, info: str, notas: List[str], frecuencias: List[float]) -> None:
        try:
            self.lbl_acorde_principal.config(text=acorde)
            self.lbl_info_acorde.config(text=info)
            
            if notas:
                self.lbl_notas_detectadas.config(text=f"Notas: {', '.join(notas)}")
                frec_str = ", ".join([f"{f:.1f}Hz" for f in frecuencias])
                self.lbl_frecuencias_detectadas.config(text=f"Frecuencias: {frec_str}")
            else:
                self.lbl_notas_detectadas.config(text="Notas: --")
                self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")
            
            self.lbl_estado.config(text=f"Acorde: {acorde}")
            
        except Exception as error:
            print(f"Error en mostrar_acorde: {error}")
            self.mostrar_silencio()

    # --- MÉTODOS PARA EFECTOS ---
    def grabar_audio(self) -> None:
        """Graba audio para aplicar efectos"""
        try:
            self.lbl_estado_grabacion.config(text="Grabando...")
            self.root.update()
            
            audio = self.procesador_efectos.grabar_audio(duracion=3.0)
            self.lbl_estado_grabacion.config(text="Grabación completada")
            
            self.actualizar_grafica_audio(self.ax_original, audio, "Audio Original")
            
        except Exception as error:
            messagebox.showerror("Error", f"Error al grabar audio: {error}")
            self.lbl_estado_grabacion.config(text="Error en grabación")

    def aplicar_efecto(self) -> None:
        """Aplica el efecto seleccionado al audio grabado"""
        try:
            efecto = self.efecto_var.get()
            intensidad = self.slider_intensidad.get()
            
            if self.procesador_efectos.audio_original is None:
                messagebox.showwarning("Advertencia", "Primero graba audio antes de aplicar efectos")
                return
            
            audio_procesado = self.procesador_efectos.aplicar_efecto(efecto, intensidad)
            self.lbl_estado_grabacion.config(text=f"Efecto aplicado: {efecto}")
            
            self.actualizar_grafica_audio(self.ax_efecto, audio_procesado, f"Audio con {efecto}")
            
        except Exception as error:
            messagebox.showerror("Error", f"Error al aplicar efecto: {error}")

    def reproducir_original(self) -> None:
        """Reproduce el audio original"""
        try:
            self.procesador_efectos.reproducir_original()
        except Exception as error:
            messagebox.showerror("Error", f"Error al reproducir audio original: {error}")

    def reproducir_con_efecto(self) -> None:
        """Reproduce el audio con efecto aplicado"""
        try:
            self.procesador_efectos.reproducir_procesado()
        except Exception as error:
            messagebox.showerror("Error", f"Error al reproducir audio con efecto: {error}")

    def guardar_resultado(self) -> None:
        """Guarda el audio procesado en un archivo"""
        try:
            if self.procesador_efectos.audio_procesado is None:
                messagebox.showwarning("Advertencia", "Primero aplica un efecto antes de guardar")
                return
            
            archivo = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
            )
            
            if archivo:
                self.procesador_efectos.guardar_audio(archivo, self.procesador_efectos.audio_procesado)
                messagebox.showinfo("Éxito", f"Audio guardado en: {archivo}")
                
        except Exception as error:
            messagebox.showerror("Error", f"Error al guardar archivo: {error}")

    def actualizar_grafica_audio(self, ax, audio_data: np.ndarray, titulo: str) -> None:
        """Actualiza las gráficas de audio"""
        try:
            ax.clear()
            ax.set_title(titulo, color='white')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3, color='white')
            
            if audio_data is not None and len(audio_data) > 0:
                muestras = min(len(audio_data), 10000)
                indices = np.linspace(0, len(audio_data)-1, muestras, dtype=int)
                ax.plot(indices, audio_data[indices], color='#3498db', lw=1)
            
            self.canvas_efectos.draw()
            
        except Exception as error:
            print(f"Error actualizando gráfica: {error}")

    def cerrar_aplicacion(self) -> None:
        """Cierra la aplicación de forma segura"""
        print("Cerrando aplicación...")
        self.detener_captura()
        sd.stop()
        if self.stream:
            try:
                self.stream.close()
                print("Stream de audio cerrado")
            except Exception as error:
                print(f"Error al cerrar stream: {error}")
        plt.close('all')
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    try:
        main_root = tk.Tk()
        app = AfinadorGuitarraAcustica(main_root)
        
        def on_closing():
            app.cerrar_aplicacion()
            
        main_root.protocol("WM_DELETE_WINDOW", on_closing)
        main_root.mainloop()
        
    except Exception as main_error:
        print(f"Error: {main_error}")