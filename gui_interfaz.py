# ============================================================
# gui_interfaz.py
# Interfaz gráfica (Tkinter) para afinador + acordes + espectro + efectos
# Usa la lógica de:
#   - core_afinador
#   - core_efectos
#   - core_esp32_serial (para enviar al LCD del ESP32)
# ============================================================

from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import time
from typing import Any, List, Optional

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Núcleo de afinador / análisis ----
from core_afinador import (
    FRECUENCIA_MUESTREO,
    TAMANO_BLOQUE,
    UMBRAL_VOLUMEN,
    ProcesadorGuitarraAcustica,
    calcular_desviacion_cents,
    obtener_frecuencia_objetivo,
)

# ---- Núcleo de efectos ----
from core_efectos import ProcesadorEfectos

# ---- Comunicación con ESP32 / LCD ----
from core_esp32_serial import enviar_lcd


class AfinadorGuitarraAcusticaGUI:
    """
    Clase principal de la GUI.

    - Modos:
        * Afinador
        * Detector de acordes
        * Visualizador de espectro
        * Pedalera de efectos

    - Envía información al ESP32 mediante enviar_lcd(...)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎸 Analizador de Audio - Fourier + Efectos")
        self.root.geometry("900x800")

        # Estado general
        self.escuchando: bool = False
        self.modo_actual: str = "afinador"
        self.stream: Optional[sd.InputStream] = None
        self.procesando_audio: bool = False

        # Núcleo de lógica
        self.procesador = ProcesadorGuitarraAcustica()
        self.procesador_efectos = ProcesadorEfectos()

        # Referencias para gráficas
        self.win_graf: Optional[tk.Toplevel] = None
        self.linea: Optional[Any] = None
        self.punto: Optional[Any] = None
        self.texto: Optional[Any] = None
        self.ax: Optional[Any] = None
        self.canvas_plot: Optional[FigureCanvasTkAgg] = None
        self.datos_graf: Optional[Any] = None

        # Espectro
        self.linea_espectro: Optional[Any] = None
        self.canvas_espectro: Optional[FigureCanvasTkAgg] = None
        self.ax_espectro: Optional[Any] = None
        self.fig_espectro: Optional[Any] = None

        # Gráficas de efectos
        self.fig = None
        self.ax_original = None
        self.ax_efecto = None
        self.canvas_efectos = None

        # Widgets de UI
        self.btn_afinador = None
        self.btn_acordes = None
        self.btn_espectro = None
        self.btn_efectos = None

        self.frame_afinador = None
        self.frame_acordes = None
        self.frame_espectro = None
        self.frame_efectos = None

        self.lbl_resultado = None
        self.lbl_frecuencia = None
        self.lbl_cuerda = None
        self.lbl_desviacion = None

        self.canvas_afinacion = None
        self.aguja = None

        self.lbl_acorde_principal = None
        self.lbl_info_acorde = None
        self.lbl_notas_detectadas = None
        self.lbl_frecuencias_detectadas = None

        self.lbl_frecuencia_principal = None
        self.lbl_volumen = None

        self.btn_grabar = None
        self.lbl_estado_grabacion = None
        self.efecto_var = None
        self.combo_efectos = None
        self.slider_intensidad = None

        self.lbl_estado = None
        self.btn_captura = None
        self.btn_detener_main = None

        # Crear UI + audio
        self.crear_interfaz()
        self.inicializar_audio()

    # ------------------------------------------------------------
    # CREAR INTERFAZ
    # ------------------------------------------------------------
    def crear_interfaz(self) -> None:
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            main_frame,
            text="🎸 ANALIZADOR DE AUDIO - FOURIER + EFECTOS",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=10)

        # ----- Menú superior -----
        menu_frame = tk.Frame(main_frame, bg="#34495e")
        menu_frame.pack(fill=tk.X, pady=5, padx=10)

        self.btn_afinador = tk.Button(
            menu_frame,
            text="🎵 AFINADOR",
            command=self.activar_modo_afinador,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
        )
        self.btn_afinador.pack(side=tk.LEFT, padx=2)

        self.btn_acordes = tk.Button(
            menu_frame,
            text="🎶 DETECTOR ACORDES",
            command=self.activar_modo_acordes,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
        )
        self.btn_acordes.pack(side=tk.LEFT, padx=2)

        self.btn_espectro = tk.Button(
            menu_frame,
            text="📈 VISUALIZAR ESPECTRO",
            command=self.activar_modo_espectro,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
        )
        self.btn_espectro.pack(side=tk.LEFT, padx=2)

        self.btn_efectos = tk.Button(
            menu_frame,
            text="🎛️ PEDALERA",
            command=self.activar_modo_efectos,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
        )
        self.btn_efectos.pack(side=tk.LEFT, padx=2)

        # ----- Frame AFINADOR -----
        self.frame_afinador = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)

        self.lbl_resultado = tk.Label(
            self.frame_afinador,
            text="--",
            font=("Arial", 32, "bold"),
            bg="#ecf0f1",
            fg="#7f8c8d",
        )
        self.lbl_resultado.pack(pady=10)

        info_frame_af = tk.Frame(self.frame_afinador, bg="#ecf0f1")
        info_frame_af.pack(pady=8)

        self.lbl_frecuencia = tk.Label(
            info_frame_af,
            text="0.00 Hz",
            font=("Arial", 11),
            bg="#ecf0f1",
        )
        self.lbl_frecuencia.pack(side=tk.LEFT, padx=10)

        self.lbl_cuerda = tk.Label(
            info_frame_af,
            text="--",
            font=("Arial", 11, "bold"),
            bg="#ecf0f1",
        )
        self.lbl_cuerda.pack(side=tk.LEFT, padx=10)

        self.lbl_desviacion = tk.Label(
            info_frame_af,
            text="±0 cents",
            font=("Arial", 10),
            bg="#ecf0f1",
        )
        self.lbl_desviacion.pack(side=tk.LEFT, padx=10)

        self.crear_indicador_afinacion(self.frame_afinador)

        # ----- Frame ACORDES -----
        self.frame_acordes = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)

        tk.Label(
            self.frame_acordes,
            text="DETECCIÓN DE ACORDES CON ANÁLISIS FOURIER",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
        ).pack(pady=10)

        self.lbl_acorde_principal = tk.Label(
            self.frame_acordes,
            text="--",
            font=("Arial", 28, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50",
        )
        self.lbl_acorde_principal.pack(pady=5)

        self.lbl_info_acorde = tk.Label(
            self.frame_acordes,
            text="Toca un acorde...",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#7f8c8d",
        )
        self.lbl_info_acorde.pack(pady=2)

        self.lbl_notas_detectadas = tk.Label(
            self.frame_acordes,
            text="Notas: --",
            font=("Arial", 9),
            bg="#ecf0f1",
            fg="#34495e",
        )
        self.lbl_notas_detectadas.pack(pady=2)

        self.lbl_frecuencias_detectadas = tk.Label(
            self.frame_acordes,
            text="Frecuencias: --",
            font=("Arial", 8),
            bg="#ecf0f1",
            fg="#95a5a6",
        )
        self.lbl_frecuencias_detectadas.pack(pady=2)

        # ----- Frame ESPECTRO -----
        self.frame_espectro = tk.Frame(main_frame, bg="#2c3e50", relief=tk.RAISED, bd=2)

        tk.Label(
            self.frame_espectro,
            text="📈 VISUALIZADOR DE ESPECTRO EN TIEMPO REAL",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="white",
        ).pack(pady=15)

        info_espectro_frame = tk.Frame(self.frame_espectro, bg="#2c3e50")
        info_espectro_frame.pack(pady=10)

        self.lbl_frecuencia_principal = tk.Label(
            info_espectro_frame,
            text="Frecuencia Principal: -- Hz",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="#00ff88",
        )
        self.lbl_frecuencia_principal.pack()

        self.lbl_volumen = tk.Label(
            info_espectro_frame,
            text="Volumen: --",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#cccccc",
        )
        self.lbl_volumen.pack()

        espectro_graf_frame = tk.Frame(self.frame_espectro, bg="#2c3e50")
        espectro_graf_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.fig_espectro, self.ax_espectro = plt.subplots(figsize=(8, 4))
        self.fig_espectro.patch.set_facecolor("#2c3e50")
        self.ax_espectro.set_facecolor("#34495e")

        self.ax_espectro.set_title(
            "ESPECTRO DE FRECUENCIAS - TIEMPO REAL",
            color="white",
            fontsize=12,
        )
        self.ax_espectro.set_xlim(50, 2000)
        self.ax_espectro.set_ylim(0, 1.0)
        self.ax_espectro.set_xlabel("Frecuencia (Hz)", color="white")
        self.ax_espectro.set_ylabel("Magnitud", color="white")
        self.ax_espectro.tick_params(colors="white")
        self.ax_espectro.grid(True, alpha=0.3, color="white")

        x_empty = np.linspace(50, 2000, 100)
        y_empty = np.zeros(100)
        (self.linea_espectro,) = self.ax_espectro.plot(
            x_empty, y_empty, color="#00ff88", lw=1.5, alpha=0.8
        )

        self.canvas_espectro = FigureCanvasTkAgg(
            self.fig_espectro, master=espectro_graf_frame
        )
        self.canvas_espectro.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ----- Frame EFECTOS -----
        self.frame_efectos = tk.Frame(main_frame, bg="#2c3e50", relief=tk.RAISED, bd=2)

        tk.Label(
            self.frame_efectos,
            text="🎛️ PEDALERA DE EFECTOS DSP",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="white",
        ).pack(pady=15)

        # Controles de grabación
        grabacion_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        grabacion_frame.pack(pady=10, fill=tk.X)

        self.btn_grabar = tk.Button(
            grabacion_frame,
            text="⏺️ GRABAR 3 SEGUNDOS",
            command=self.grabar_audio,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            width=20,
        )
        self.btn_grabar.pack(side=tk.LEFT, padx=10)

        self.lbl_estado_grabacion = tk.Label(
            grabacion_frame,
            text="No grabado",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#ecf0f1",
        )
        self.lbl_estado_grabacion.pack(side=tk.LEFT, padx=10)

        # Selector de efectos
        efectos_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        efectos_frame.pack(pady=10, fill=tk.X)

        tk.Label(
            efectos_frame,
            text="Efecto:",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=10)

        self.efecto_var = tk.StringVar()
        opciones = ["Filtro Pasa-Bajos", "Distorsión", "Modulación", "Delay", "Reverb"]
        self.combo_efectos = ttk.Combobox(
            efectos_frame,
            textvariable=self.efecto_var,
            values=opciones,
            state="readonly",
            width=20,
        )
        self.combo_efectos.current(0)
        self.combo_efectos.pack(side=tk.LEFT, padx=10)

        tk.Label(
            efectos_frame,
            text="Intensidad:",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=10)

        self.slider_intensidad = tk.Scale(
            efectos_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=150,
            bg="#34495e",
            fg="white",
            highlightbackground="#2c3e50",
        )
        self.slider_intensidad.set(50)
        self.slider_intensidad.pack(side=tk.LEFT, padx=10)

        self.btn_aplicar_efecto = tk.Button(
            efectos_frame,
            text="⚡ APLICAR EFECTO",
            command=self.aplicar_efecto,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.btn_aplicar_efecto.pack(side=tk.LEFT, padx=10)

        # Controles de reproducción
        reproduccion_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        reproduccion_frame.pack(pady=15, fill=tk.X)

        btn_repro_original = tk.Button(
            reproduccion_frame,
            text="▶ REPRODUCIR ORIGINAL",
            command=self.reproducir_original,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10),
            width=18,
        )
        btn_repro_original.pack(side=tk.LEFT, padx=10)

        btn_repro_efecto = tk.Button(
            reproduccion_frame,
            text="▶ REPRODUCIR CON EFECTO",
            command=self.reproducir_con_efecto,
            bg="#e67e22",
            fg="white",
            font=("Arial", 10),
            width=20,
        )
        btn_repro_efecto.pack(side=tk.LEFT, padx=10)

        btn_guardar = tk.Button(
            reproduccion_frame,
            text="💾 GUARDAR RESULTADO",
            command=self.guardar_resultado,
            bg="#3498db",
            fg="white",
            font=("Arial", 10),
        )
        btn_guardar.pack(side=tk.LEFT, padx=10)

        btn_detener = tk.Button(
            reproduccion_frame,
            text="⏹ DETENER",
            command=sd.stop,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10),
        )
        btn_detener.pack(side=tk.LEFT, padx=10)

        # Gráficas de efectos
        graficas_frame = tk.Frame(self.frame_efectos, bg="#2c3e50")
        graficas_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_original, self.ax_efecto) = plt.subplots(2, 1, figsize=(8, 4))
        self.fig.patch.set_facecolor("#2c3e50")

        for ax in (self.ax_original, self.ax_efecto):
            ax.set_facecolor("#34495e")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for side in ["bottom", "top", "left", "right"]:
                ax.spines[side].set_color("white")
            ax.grid(True, alpha=0.3, color="white")

        self.ax_original.set_title("Audio Original", color="white")
        self.ax_original.set_ylim(-1.1, 1.1)
        self.ax_efecto.set_title("Audio con Efecto", color="white")
        self.ax_efecto.set_ylim(-1.1, 1.1)

        self.canvas_efectos = FigureCanvasTkAgg(self.fig, master=graficas_frame)
        self.canvas_efectos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- Estado general + botones captura ----
        self.lbl_estado = tk.Label(
            main_frame,
            text="Selecciona un modo y presiona INICIAR",
            font=("Arial", 10),
        )
        self.lbl_estado.pack(pady=8)

        controles_frame = tk.Frame(main_frame)
        controles_frame.pack(pady=10)

        self.btn_captura = tk.Button(
            controles_frame,
            text="🎙️ INICIAR CAPTURA",
            command=self.toggle_captura,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15,
            height=1,
        )
        self.btn_captura.pack(side=tk.LEFT, padx=5)

        self.btn_detener_main = tk.Button(
            controles_frame,
            text="🛑 DETENER",
            command=self.detener_captura,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            width=12,
            height=1,
        )
        self.btn_detener_main.pack(side=tk.LEFT, padx=5)

        # Mostrar afinador por defecto
        self.activar_modo_afinador()

    # ------------------------------------------------------------
    # INDICADOR DE AFINACIÓN
    # ------------------------------------------------------------
    def crear_indicador_afinacion(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg="#ecf0f1")
        frame.pack(pady=10)

        self.canvas_afinacion = tk.Canvas(frame, width=300, height=60, bg="#2c3e50")
        self.canvas_afinacion.pack()

        for cents in [-50, -25, 0, 25, 50]:
            x = 150 + cents
            color = "#e74c3c" if cents == 0 else "#7f8c8d"
            self.canvas_afinacion.create_line(x, 20, x, 40, fill=color, width=2)
            self.canvas_afinacion.create_text(
                x,
                50,
                text=str(cents),
                fill="white",
                font=("Arial", 8),
            )

        self.aguja = self.canvas_afinacion.create_line(
            150, 15, 150, 45, fill="#3498db", width=3
        )

    # ------------------------------------------------------------
    # MODOS
    # ------------------------------------------------------------
    def activar_modo_afinador(self) -> None:
        self.modo_actual = "afinador"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_afinador.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(
            text="Modo Afinador - Toca una cuerda individual de guitarra"
        )

        # Enviar info al LCD (tipo 'A')
        enviar_lcd("A", "AFINADOR", "Toca una cuerda", "", "Fila 4 = Volver")

    def activar_modo_acordes(self) -> None:
        self.modo_actual = "acordes"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_acordes.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(text="Modo Acordes - Toca un acorde completo de guitarra")

        enviar_lcd("A", "ACORDES", "Toca un acorde", "", "Fila 4 = Volver")

    def activar_modo_espectro(self) -> None:
        self.modo_actual = "espectro"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_espectro.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(
            text="Modo Espectro - Visualizando espectro de frecuencias en tiempo real"
        )

        enviar_lcd("A", "ESPECTRO", "Analizando", "", "Fila 4 = Volver")

    def activar_modo_efectos(self) -> None:
        self.modo_actual = "efectos"
        self.actualizar_botones_menu()
        self.ocultar_todos_frames()
        self.frame_efectos.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.lbl_estado.config(
            text="Modo Efectos - Graba audio y aplica efectos DSP"
        )

        enviar_lcd("E", "EFECTOS DSP", "Graba y aplica FX", "", "Fila 4 = Volver")

    def actualizar_botones_menu(self) -> None:
        botones = {
            "afinador": self.btn_afinador,
            "acordes": self.btn_acordes,
            "espectro": self.btn_espectro,
            "efectos": self.btn_efectos,
        }

        for modo, boton in botones.items():
            if modo == self.modo_actual:
                boton.config(bg="#3498db")
            else:
                boton.config(bg="#95a5a6")

    def ocultar_todos_frames(self) -> None:
        for frame in (
            self.frame_afinador,
            self.frame_acordes,
            self.frame_espectro,
            self.frame_efectos,
        ):
            frame.pack_forget()

    # ------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------
    def inicializar_audio(self) -> None:
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=FRECUENCIA_MUESTREO,
                blocksize=TAMANO_BLOQUE,
                latency="low",
            )
            print("✅ Sistema de audio inicializado")
        except Exception as error:
            print(f"Error audio: {error}")

    def audio_callback(self, indata, frames, callback_time, status) -> None:
        if not self.escuchando or self.procesando_audio:
            return

        self.procesando_audio = True
        try:
            audio_data = indata[:, 0]
            audio_flat = audio_data.flatten() if audio_data is not None else np.array([])
            volumen = (
                float(np.sqrt(np.mean(audio_flat**2)))
                if len(audio_flat) > 0
                else 0.0
            )

            # ----- Modo espectro -----
            if self.modo_actual == "espectro" and len(audio_flat) > 0:
                try:
                    f, magnitudes = self.procesador.detector_notas.calcular_espectro_mejorado(
                        audio_flat
                    )
                    if len(f) > 0 and len(magnitudes) > 0 and self.linea_espectro is not None:
                        mascara = (f >= 50) & (f <= 2000)
                        if np.any(mascara):
                            idx_principal = np.argmax(magnitudes[mascara])
                            freq_principal = f[mascara][idx_principal]

                            self.safe_after(
                                lambda: self.lbl_frecuencia_principal.config(
                                    text=f"Frecuencia Principal: {freq_principal:.1f} Hz"
                                )
                            )
                            self.safe_after(
                                lambda: self.lbl_volumen.config(
                                    text=f"Volumen: {volumen:.3f}"
                                )
                            )

                        magnitudes_norm = (
                            magnitudes / np.max(magnitudes)
                            if np.max(magnitudes) > 0
                            else magnitudes
                        )
                        self.linea_espectro.set_data(f, magnitudes_norm)
                        if self.canvas_espectro is not None:
                            self.canvas_espectro.draw_idle()

                except Exception as spec_error:
                    print(f"Error en visualizador espectro: {spec_error}")

            # Silencio
            if volumen < UMBRAL_VOLUMEN:
                self.safe_after(self.mostrar_silencio)
                return

            # ----- Modo afinador -----
            if self.modo_actual == "afinador" and len(audio_flat) > 0:
                frecuencia, cuerda = self.procesador.detectar_fundamental_segura(
                    audio_flat
                )
                if frecuencia and cuerda:
                    freq_estable, cuerda_estable = self.procesador.procesar_estabilidad_segura(
                        frecuencia, cuerda
                    )
                    if freq_estable and cuerda_estable:
                        self.safe_after(
                            lambda: self.mostrar_afinacion(freq_estable, cuerda_estable)
                        )
                else:
                    self.safe_after(self.mostrar_silencio)

            # ----- Modo acordes -----
            elif self.modo_actual == "acordes" and len(audio_flat) > 0:
                acorde, info = self.procesador.detectar_acorde_completo(audio_flat)
                notas_detectadas = self.procesador.detector_notas.detectar_notas_en_audio(
                    audio_flat
                )
                notas_nombres = [
                    n["nota"] for n in notas_detectadas if n["nota"] is not None
                ]
                frecuencias = [n["frecuencia"] for n in notas_detectadas]

                self.safe_after(
                    lambda: self.mostrar_acorde(
                        acorde, info, notas_nombres, frecuencias
                    )
                )

        except Exception as error:
            print(f"Error en callback: {error}")
            self.safe_after(self.mostrar_silencio)
        finally:
            self.procesando_audio = False

    def safe_after(self, func) -> None:
        try:
            if self.root and hasattr(self.root, "winfo_exists") and self.root.winfo_exists():
                self.root.after(0, func)
        except Exception as error:
            print(f"Error en safe_after: {error}")

    # ------------------------------------------------------------
    # CONTROL CAPTURA
    # ------------------------------------------------------------
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
                    "espectro": "Analizando espectro",
                    "efectos": "Modo pedalera",
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
        self.escuchando = False
        if self.stream:
            try:
                self.stream.stop()
            except Exception as error:
                print(f"Error al detener stream: {error}")
        self.btn_captura.config(text="🎙️ INICIAR", bg="#27ae60")
        self.lbl_estado.config(text="Captura detenida")
        self.mostrar_silencio()

    # ------------------------------------------------------------
    # ESTADOS DE PANTALLA
    # ------------------------------------------------------------
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
                if (
                    not self.procesador.detector_notas.ultimo_acorde_detectado
                    or tiempo_actual - ultimo_tiempo >= 2.0
                ):
                    self.lbl_acorde_principal.config(text="--")
                    self.lbl_info_acorde.config(text="Toca un acorde...")
                    self.lbl_notas_detectadas.config(text="Notas: --")
                    self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")

            elif self.modo_actual == "espectro":
                self.lbl_frecuencia_principal.config(
                    text="Frecuencia Principal: -- Hz"
                )
                self.lbl_volumen.config(text="Volumen: --")
                if self.linea_espectro is not None:
                    x_empty = np.linspace(50, 2000, 100)
                    y_empty = np.zeros(100)
                    self.linea_espectro.set_data(x_empty, y_empty)
                    if self.canvas_espectro is not None:
                        self.canvas_espectro.draw_idle()

            self.lbl_estado.config(
                text="Toca tu guitarra o habla al micrófono..."
            )
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

            # Enviar info al LCD (modo afinador, tipo 'A')
            linea1 = f"{cuerda} {frecuencia:.1f}Hz"
            linea2 = accion
            linea3 = f"{cents:+.1f} cents"
            linea4 = "Fila 4 = Volver"
            enviar_lcd("A", linea1, linea2, linea3, linea4)

        except Exception as error:
            print(f"Error en mostrar_afinacion: {error}")
            self.mostrar_silencio()

    def mostrar_acorde(
        self,
        acorde: str,
        info: str,
        notas: List[str],
        frecuencias: List[float],
    ) -> None:
        try:
            self.lbl_acorde_principal.config(text=acorde)
            self.lbl_info_acorde.config(text=info)

            if notas:
                self.lbl_notas_detectadas.config(text=f"Notas: {', '.join(notas)}")
                frec_str = ", ".join([f"{f:.1f}Hz" for f in frecuencias])
                self.lbl_frecuencias_detectadas.config(
                    text=f"Frecuencias: {frec_str}"
                )
            else:
                self.lbl_notas_detectadas.config(text="Notas: --")
                self.lbl_frecuencias_detectadas.config(text="Frecuencias: --")

            self.lbl_estado.config(text=f"Acorde: {acorde}")

            # Enviar info de acordes al LCD
            linea1 = f"Acorde: {acorde}"[:20]
            linea2 = info[:20]
            linea3 = (
                f"Notas: {' '.join(notas)[:20]}" if notas else ""
            )
            linea4 = "Fila 4 = Volver"
            enviar_lcd("A", linea1, linea2, linea3, linea4)

        except Exception as error:
            print(f"Error en mostrar_acorde: {error}")
            self.mostrar_silencio()

    # ------------------------------------------------------------
    # EFECTOS
    # ------------------------------------------------------------
    def grabar_audio(self) -> None:
        try:
            self.lbl_estado_grabacion.config(text="Grabando...")
            self.root.update()

            audio = self.procesador_efectos.grabar_audio(duracion=3.0)
            self.lbl_estado_grabacion.config(text="Grabación completada")

            self.actualizar_grafica_audio(
                self.ax_original, audio, "Audio Original"
            )

        except Exception as error:
            messagebox.showerror("Error", f"Error al grabar audio: {error}")
            self.lbl_estado_grabacion.config(text="Error en grabación")

    def aplicar_efecto(self) -> None:
        try:
            efecto = self.efecto_var.get()
            intensidad = self.slider_intensidad.get()

            if self.procesador_efectos.audio_original is None:
                messagebox.showwarning(
                    "Advertencia",
                    "Primero graba audio antes de aplicar efectos",
                )
                return

            audio_procesado = self.procesador_efectos.aplicar_efecto(
                efecto, intensidad
            )
            self.lbl_estado_grabacion.config(
                text=f"Efecto aplicado: {efecto}"
            )

            self.actualizar_grafica_audio(
                self.ax_efecto, audio_procesado, f"Audio con {efecto}"
            )

        except Exception as error:
            messagebox.showerror("Error", f"Error al aplicar efecto: {error}")

    def reproducir_original(self) -> None:
        try:
            self.procesador_efectos.reproducir_original()
        except Exception as error:
            messagebox.showerror(
                "Error", f"Error al reproducir audio original: {error}"
            )

    def reproducir_con_efecto(self) -> None:
        try:
            self.procesador_efectos.reproducir_procesado()
        except Exception as error:
            messagebox.showerror(
                "Error", f"Error al reproducir audio con efecto: {error}"
            )

    def guardar_resultado(self) -> None:
        try:
            if self.procesador_efectos.audio_procesado is None:
                messagebox.showwarning(
                    "Advertencia",
                    "Primero aplica un efecto antes de guardar",
                )
                return

            archivo = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[
                    ("Archivos WAV", "*.wav"),
                    ("Todos los archivos", "*.*"),
                ],
            )

            if archivo:
                self.procesador_efectos.guardar_audio(
                    archivo, self.procesador_efectos.audio_procesado
                )
                messagebox.showinfo(
                    "Éxito", f"Audio guardado en: {archivo}"
                )

        except Exception as error:
            messagebox.showerror(
                "Error", f"Error al guardar archivo: {error}"
            )

    def actualizar_grafica_audio(self, ax, audio_data, titulo: str) -> None:
        try:
            ax.clear()
            ax.set_title(titulo, color="white")
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3, color="white")

            if audio_data is not None and len(audio_data) > 0:
                muestras = min(len(audio_data), 10000)
                indices = np.linspace(0, len(audio_data) - 1, muestras, dtype=int)
                ax.plot(indices, audio_data[indices], color="#3498db", lw=1)

            if self.canvas_efectos:
                self.canvas_efectos.draw()

        except Exception as error:
            print(f"Error actualizando gráfica: {error}")

    # ------------------------------------------------------------
    # INTEGRACIÓN CON ESP32 (opcional desde main)
    # ------------------------------------------------------------
    def on_esp32_data(self, data: str) -> None:
        """
        Callback opcional para usar con ESP32Bridge.agregar_listener.
        main.py puede hacer:
            esp32.agregar_listener(app.on_esp32_data)
        """
        if data == "MODO_AFINADOR":
            self.root.after(0, self.activar_modo_afinador)
        elif data == "MODO_EFECTOS":
            self.root.after(0, self.activar_modo_efectos)
        elif data == "INICIAR_AFINADOR":
            self.root.after(0, self.toggle_captura)
        elif data == "DETENER_AFINADOR":
            self.root.after(0, self.detener_captura)

            
            

    # ------------------------------------------------------------
    # CIERRE
    # ------------------------------------------------------------
    def cerrar_aplicacion(self) -> None:
        print("Cerrando aplicación...")
        self.detener_captura()
        sd.stop()
        try:
            if self.stream:
                self.stream.close()
        except Exception as error:
            print(f"Error al cerrar stream: {error}")
        plt.close("all")
        self.root.quit()
        self.root.destroy()
