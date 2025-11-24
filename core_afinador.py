# core_afinador.py

from __future__ import annotations

import math
import time
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional, Deque, Any

import numpy as np
from scipy.signal import stft, find_peaks

# ============================
# PARÁMETROS DE AUDIO
# ============================

FRECUENCIA_MUESTREO = 44100
TAMANO_BLOQUE = 4096
NPERSEG = 8192
NOVERLAP = 6144
UMBRAL_VOLUMEN = 0.001

# ============================
# TABLAS MUSICALES
# ============================

# Frecuencias aproximadas de notas (base)
NOTAS_MUSICALES: Dict[str, float] = {
    "C": 130.81, "C#": 138.59, "D": 146.83, "D#": 155.56,
    "E": 164.81, "F": 174.61, "F#": 185.00, "G": 196.00,
    "G#": 207.65, "A": 220.00, "A#": 233.08, "B": 246.94,
}

# Frecuencias reales de cuerdas de guitarra + rango válido aproximado
#  (frecuencia_objetivo, freq_min, freq_max)
CUERDAS_GUITARRA: Dict[str, Tuple[float, int, int]] = {
    "E2": (161.5, 150, 175),
    "A2": (107.7, 95, 125),
    "D3": (145.3, 130, 155),
    "G3": (199.2, 180, 220),
    "B3": (247.6, 230, 270),
    "E4": (328.4, 300, 350)
}

# Acordes y notas componentes (solo nombres de nota, sin octava)
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


# =========================================
#  ANALIZADOR DE FOURIER (SERIES)
# =========================================

class AnalizadorFourier:
    """
    Implementa las Series de Fourier:
    f(t) = a0 + Σ [an cos(n·ω0·t) + bn sin(n·ω0·t)]
    """

    def __init__(self, num_armonicos: int = 10):
        self.num_armonicos = num_armonicos

    def calcular_series_fourier(
        self,
        audio_data: np.ndarray,
        frecuencia_fundamental: float
    ) -> Dict[str, Any]:
        """
        Calcula coeficientes a0, an, bn y la magnitud de cada armónico.
        """
        try:
            N = len(audio_data)
            if N == 0:
                return {}

            T = N / FRECUENCIA_MUESTREO
            t = np.linspace(0, T, N, endpoint=False)
            omega_0 = 2 * math.pi * frecuencia_fundamental

            a0 = float(np.mean(audio_data))

            coeficientes = {
                "a0": a0,
                "a_n": [],
                "b_n": [],
                "armonicos": [],
                "magnitudes": [],
                "frecuencia_fundamental": frecuencia_fundamental
            }

            for n in range(1, self.num_armonicos + 1):
                cos_term = np.cos(n * omega_0 * t)
                sin_term = np.sin(n * omega_0 * t)

                an_val = 2.0 / T * np.trapz(audio_data * cos_term, t)
                bn_val = 2.0 / T * np.trapz(audio_data * sin_term, t)

                an_val = float(an_val)
                bn_val = float(bn_val)
                magnitud = math.sqrt(an_val**2 + bn_val**2)

                coeficientes["a_n"].append(an_val)
                coeficientes["b_n"].append(bn_val)
                coeficientes["armonicos"].append(n)
                coeficientes["magnitudes"].append(float(magnitud))

            return coeficientes

        except Exception as error:
            print(f"Error en series de Fourier: {error}")
            return {}

    def analizar_timbre(self, coeficientes: Dict[str, Any]) -> Dict[str, float]:
        """
        Devuelve características espectrales básicas del timbre:
        - riqueza_espectral
        - armonico_principal
        - relacion_fundamental
        - brillantez
        """
        try:
            if not coeficientes or "magnitudes" not in coeficientes:
                return {}

            magnitudes = coeficientes["magnitudes"]
            if not magnitudes:
                return {}

            magnitud_total = sum(magnitudes)
            if magnitud_total <= 0:
                return {}

            magnitud_fundamental = magnitudes[0]

            riqueza_espectral = sum(magnitudes[1:5]) / magnitud_total
            armonico_principal = int(np.argmax(magnitudes) + 1)
            relacion_fundamental = magnitud_fundamental / magnitud_total
            brillantez = sum(magnitudes[3:]) / magnitud_total

            return {
                "riqueza_espectral": float(riqueza_espectral),
                "armonico_principal": armonico_principal,
                "relacion_fundamental": float(relacion_fundamental),
                "brillantez": float(brillantez),
            }

        except Exception as error:
            print(f"Error analizando timbre: {error}")
            return {}


# =========================================
#  DETECTOR DE NOTAS Y ACORDES
# =========================================

class DetectorNotasAcordes:
    def __init__(self):
        self.historico_acordes: Deque[str] = deque(maxlen=5)
        self.historico_notas: Deque[str] = deque(maxlen=10)

        self.ultimo_acorde_detectado: Optional[str] = None
        self.tiempo_ultimo_acorde: float = 0.0

        self.analizador_fourier = AnalizadorFourier()

    # -------------------------
    #   UTILIDADES
    # -------------------------

    def frecuencia_a_nota(self, frecuencia: float) -> Tuple[Optional[str], float]:
        """
        Convierte una frecuencia a la nota musical más cercana.
        Devuelve (nota, error_log2).
        """
        mejor_nota: Optional[str] = None
        menor_error: float = float("inf")

        if frecuencia <= 0:
            return None, menor_error

        for nota, freq_base in NOTAS_MUSICALES.items():
            for octava in range(2, 6):
                freq_octava = freq_base * (2 ** (octava - 4))
                if freq_octava <= 0:
                    continue

                error = abs(math.log2(frecuencia / freq_octava))
                if error < menor_error:
                    menor_error = error
                    mejor_nota = nota

        return mejor_nota, menor_error

    def calcular_espectro_mejorado(
        self, audio_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        STFT con parámetros grandes (NPERSEG, NOVERLAP) para mejor resolución
        en frecuencias bajas (guitarra).
        """
        try:
            if len(audio_data) < NPERSEG:
                audio_data = np.pad(
                    audio_data, (0, NPERSEG - len(audio_data)), mode="constant"
                )

            f, _, Zxx = stft(
                audio_data,
                fs=FRECUENCIA_MUESTREO,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                window="hann",
                nfft=NPERSEG * 2,
            )

            magnitudes = np.sqrt(np.mean(np.abs(Zxx) ** 2, axis=1))
            return f, magnitudes

        except Exception as error:
            print(f"Error en cálculo de espectro: {error}")
            return np.array([]), np.array([])

    # -------------------------
    #   DETECCIÓN DE NOTAS
    # -------------------------

    def calcular_confianza(
        self, timbre: Dict[str, float], error_frecuencia: float
    ) -> float:
        """
        Combina error en frecuencia + info de timbre para dar una confianza [0,1].
        """
        confianza = 1.0 - min(error_frecuencia * 10.0, 1.0)

        if timbre:
            if timbre.get("riqueza_espectral", 0.0) > 0.3:
                confianza += 0.2
            if timbre.get("relacion_fundamental", 0.0) > 0.4:
                confianza += 0.1

        return float(min(confianza, 1.0))

    def detectar_notas_en_audio(
        self, audio_data: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de notas detectadas con:
          - nota
          - frecuencia
          - error
          - timbre (características espectrales)
          - confianza
        """
        try:
            f, magnitudes = self.calcular_espectro_mejorado(audio_data)
            if len(f) == 0 or len(magnitudes) == 0:
                return []

            notas_detectadas: List[Dict[str, Any]] = []

            umbral = 0.015 * np.max(magnitudes)
            picos, propiedades = find_peaks(
                magnitudes,
                height=umbral,
                distance=8,
                prominence=umbral * 0.5,
            )

            if len(picos) > 0 and "peak_heights" in propiedades:
                for i, idx in enumerate(picos):
                    if idx >= len(f):
                        continue

                    freq = float(f[idx])
                    if not (80.0 <= freq <= 450.0):
                        continue

                    nota, error = self.frecuencia_a_nota(freq)
                    if nota is None or error >= 0.08:
                        continue

                    magnitud_pico = float(propiedades["peak_heights"][i])
                    if magnitud_pico <= umbral * 1.2:
                        continue

                    coef_fourier = self.analizador_fourier.calcular_series_fourier(
                        audio_data, freq
                    )
                    timbre = self.analizador_fourier.analizar_timbre(coef_fourier)
                    confianza = self.calcular_confianza(timbre, error)

                    notas_detectadas.append(
                        {
                            "nota": nota,
                            "frecuencia": freq,
                            "error": error,
                            "magnitud": magnitud_pico,
                            "coeficientes_fourier": coef_fourier,
                            "timbre": timbre,
                            "confianza": confianza,
                        }
                    )

            # Ordenar por confianza y quedarnos con una por nota
            notas_detectadas.sort(key=lambda x: x["confianza"], reverse=True)

            notas_unicas: Dict[str, Dict[str, Any]] = {}
            for info in notas_detectadas:
                n = info["nota"]
                if n is None:
                    continue
                if n not in notas_unicas or info["confianza"] > notas_unicas[n]["confianza"]:
                    notas_unicas[n] = info

            return list(notas_unicas.values())

        except Exception as error:
            print(f"Error detectando notas: {error}")
            return []

    # -------------------------
    #   DETECCIÓN DE ACORDES
    # -------------------------

    def identificar_tonica_probable(self, notas: List[str]) -> str:
        """
        Dado un conjunto de notas (sin octava), devuelve una tónica probable.
        """
        if not notas:
            return ""

        notas_base = [n[0] for n in notas if n]
        if not notas_base:
            return ""

        conteo = Counter(notas_base)
        tonicas_comunes = ["A", "B", "C", "D", "E", "F", "G"]

        for t in tonicas_comunes:
            if t in conteo:
                return t

        return conteo.most_common(1)[0][0]

    def identificar_acorde_por_notas(
        self, notas_detectadas: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Recibe la salida de detectar_notas_en_audio()
        y devuelve (nombre_acorde, info_textual).
        """
        if len(notas_detectadas) < 2:
            return "No detectado", "Muy pocas notas"

        notas_completas: List[str] = []
        info_fourier: List[Dict[str, Any]] = []

        for nd in notas_detectadas:
            nota = nd.get("nota")
            if nota is None:
                continue
            notas_completas.append(nota)
            info_fourier.append(
                {
                    "nota": nota,
                    "frecuencia": nd.get("frecuencia", 0.0),
                    "confianza": nd.get("confianza", 0.5),
                    "timbre": nd.get("timbre", {}),
                }
            )

        if not notas_completas:
            return "No detectado", "No se detectaron notas válidas"

        notas_set = set(notas_completas)
        print(f"Notas detectadas (acordes): {notas_set}")

        fourier_info_str = ", ".join(
            f"{n['nota']}({n['confianza']:.2f})" for n in info_fourier
        )
        print(f"Análisis Fourier (acordes): {fourier_info_str}")

        mejor_acorde: Optional[str] = None
        mejor_puntaje: float = 0.0
        tonica_probable = self.identificar_tonica_probable(notas_completas)

        for acorde, info in ACORDES_GUITARRA.items():
            notas_acorde = info["notas"]
            notas_acorde_set = set(notas_acorde)

            # Coincidencias exactas
            coincidencias_exactas = notas_set.intersection(notas_acorde_set)
            puntaje_exacto = len(coincidencias_exactas) / len(notas_acorde)

            # Coincidencias por letra (sin #)
            notas_base_detectadas = set(n[0] for n in notas_completas if n)
            notas_base_acorde = set(n[0] for n in notas_acorde)
            coincidencias_base = notas_base_detectadas.intersection(notas_base_acorde)
            puntaje_base = len(coincidencias_base) / len(notas_acorde)

            puntaje = puntaje_exacto * 0.7 + puntaje_base * 0.3

            tonica_acorde = notas_acorde[0]
            if tonica_acorde in notas_set:
                for nd in info_fourier:
                    if nd["nota"] == tonica_acorde:
                        puntaje += 0.3 + 0.2 * nd["confianza"]
                        break
            elif tonica_acorde[0] in notas_base_detectadas and tonica_probable == tonica_acorde[0]:
                puntaje += 0.2

            # Tercera
            tercera_acorde = notas_acorde[1]
            if tercera_acorde in notas_set:
                for nd in info_fourier:
                    if nd["nota"] == tercera_acorde:
                        puntaje += 0.3 + 0.1 * nd["confianza"]
                        break

            # Quinta
            quinta_acorde = notas_acorde[2]
            if quinta_acorde in notas_set:
                puntaje += 0.2

            # Penalizaciones específicas
            if acorde == "B Mayor" and "D" in notas_set and "D#" not in notas_set:
                puntaje -= 0.3
            elif acorde == "B menor" and "D#" in notas_set and "D" not in notas_set:
                puntaje -= 0.3
            elif acorde == "G Mayor" and "F#" in notas_set and len(notas_set) < 3:
                puntaje -= 0.2
            elif acorde == "A menor" and "C#" in notas_set:
                puntaje -= 0.4

            print(f"  {acorde}: exactas={coincidencias_exactas} ({puntaje:.1%})")

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_acorde = acorde

        tiempo_actual = time.time()

        if mejor_acorde and mejor_puntaje > 0.7:
            self.ultimo_acorde_detectado = mejor_acorde
            self.tiempo_ultimo_acorde = tiempo_actual
            self.historico_acordes.append(mejor_acorde)

            if len(self.historico_acordes) >= 3:
                conteo = Counter(self.historico_acordes)
                acorde_estable, count = conteo.most_common(1)[0]
                if count >= 2:
                    return acorde_estable, f"✅ {acorde_estable} ({mejor_puntaje:.0%})"

            if mejor_puntaje > 0.85:
                return mejor_acorde, f"🎵 {mejor_acorde} ({mejor_puntaje:.0%})"
            elif mejor_puntaje > 0.75:
                return f"Posible {mejor_acorde}", f"{mejor_puntaje:.0%} coincidencia"

        elif (
            self.ultimo_acorde_detectado
            and tiempo_actual - self.tiempo_ultimo_acorde < 2.0
        ):
            return (
                self.ultimo_acorde_detectado,
                f"⏹ {self.ultimo_acorde_detectado} (mantenido)",
            )

        return "No identificado", f"Notas: {', '.join(notas_set)}"


# =========================================
#  PROCESADOR PRINCIPAL (CUERDA / ACORDE)
# =========================================

class ProcesadorGuitarraAcustica:
    """
    - Detecta frecuencia fundamental de la guitarra
    - Identifica la cuerda
    - Aplica filtro de estabilidad en el tiempo
    - Detecta acordes completos
    """

    def __init__(self):
        self.historico_fundamentales: Deque[float] = deque(maxlen=5)
        self.historico_cuerdas: Deque[str] = deque(maxlen=5)
        self.detector_notas = DetectorNotasAcordes()

    def detectar_fundamental_segura(
        self, audio_data: np.ndarray
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Usa el espectro para encontrar la frecuencia fundamental
        en el rango de guitarra y asignar cuerda aproximada.
        """
        try:
            f, magnitudes = self.detector_notas.calcular_espectro_mejorado(audio_data)
            if len(f) == 0 or len(magnitudes) == 0:
                return None, None

            mascara = (f >= 90.0) & (f <= 380.0)
            if not np.any(mascara):
                return None, None

            f_guitarra = f[mascara]
            mag_guitarra = magnitudes[mascara]

            if len(mag_guitarra) < 3:
                return None, None

            altura_minima = 0.05 * np.max(mag_guitarra)
            picos, propiedades = find_peaks(
                mag_guitarra,
                height=altura_minima,
                distance=3,
                prominence=altura_minima * 0.3,
            )

            if len(picos) == 0 or "peak_heights" not in propiedades:
                return None, None

            idx_principal = picos[np.argmax(propiedades["peak_heights"])]
            if idx_principal >= len(f_guitarra):
                return None, None

            frecuencia = float(f_guitarra[idx_principal])
            cuerda = self.identificar_cuerda_segura(frecuencia)
            return frecuencia, cuerda

        except Exception as error:
            print(f"Error en detección fundamental: {error}")
            return None, None

    def identificar_cuerda_segura(self, frecuencia: float) -> Optional[str]:
        """
        Decide qué cuerda es más probable, usando el rango [min,max] y
        la distancia logarítmica a la frecuencia objetivo.
        """
        try:
            mejor_cuerda: Optional[str] = None
            menor_distancia: float = float("inf")

            # Primero, cuerdas cuyo rango contiene la frecuencia
            for cuerda, (f_obj, f_min, f_max) in CUERDAS_GUITARRA.items():
                if f_min <= frecuencia <= f_max and f_obj > 0:
                    dist = abs(math.log2(frecuencia / f_obj))
                    if dist < menor_distancia:
                        menor_distancia = dist
                        mejor_cuerda = cuerda

            # Si no cae en ningún rango, escoger la más cercana globalmente
            if mejor_cuerda is None:
                for cuerda, (f_obj, _, _) in CUERDAS_GUITARRA.items():
                    if f_obj <= 0:
                        continue
                    dist = abs(math.log2(frecuencia / f_obj))
                    if dist < menor_distancia:
                        menor_distancia = dist
                        mejor_cuerda = cuerda

            return mejor_cuerda

        except Exception:
            return None

    def procesar_estabilidad_segura(
        self,
        frecuencia: Optional[float],
        cuerda: Optional[str],
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Suaviza la detección combinando varias muestras:
        si la misma cuerda aparece la mayoría de las veces,
        devuelve la frecuencia mediana de esa cuerda.
        """
        try:
            if frecuencia and cuerda:
                self.historico_fundamentales.append(frecuencia)
                self.historico_cuerdas.append(cuerda)

                if len(self.historico_fundamentales) >= 3:
                    conteo = Counter(self.historico_cuerdas)
                    if not conteo:
                        return frecuencia, cuerda

                    cuerda_estable, count = conteo.most_common(1)[0]
                    if count >= len(self.historico_cuerdas) * 0.6:
                        frecs_filtradas = [
                            f
                            for f, c in zip(
                                self.historico_fundamentales,
                                self.historico_cuerdas,
                            )
                            if c == cuerda_estable
                        ]
                        if frecs_filtradas:
                            freq_est = float(np.median(frecs_filtradas))
                            return freq_est, cuerda_estable

            return frecuencia, cuerda

        except Exception as error:
            print(f"Error en estabilidad: {error}")
            return frecuencia, cuerda

    def detectar_acorde_completo(self, audio_data: np.ndarray) -> Tuple[str, str]:
        """
        Atajo para: detectar_notas_en_audio -> identificar_acorde_por_notas
        """
        notas_detectadas = self.detector_notas.detectar_notas_en_audio(audio_data)
        return self.detector_notas.identificar_acorde_por_notas(notas_detectadas)


# =========================================
#   FUNCIONES AUXILIARES
# =========================================

def calcular_desviacion_cents(f_medida: float, f_objetivo: float) -> float:
    """
    Calcula la desviación en cents entre dos frecuencias.
    """
    if f_medida <= 0 or f_objetivo <= 0:
        return 0.0
    try:
        return float(1200.0 * math.log2(f_medida / f_objetivo))
    except Exception:
        return 0.0


def obtener_frecuencia_objetivo(cuerda: str) -> float:
    """
    Devuelve la frecuencia objetivo (central) de una cuerda tipo 'E2', 'A2', etc.
    """
    if cuerda in CUERDAS_GUITARRA:
        return float(CUERDAS_GUITARRA[cuerda][0])
    return 0.0
