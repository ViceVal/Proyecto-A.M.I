# ============================================================
# core/esp32_serial.py
# Manejo de comunicación por puerto serie con ESP32
# ============================================================

import serial
import time
import threading

# Ajusta según tu PC
PORT = "/dev/ttyUSB0"
BAUD = 115200


class ESP32Bridge:
    """
    Puente de comunicación serial con ESP32.
    - Conectar
    - Enviar mensajes
    - Escuchar mensajes en hilo
    - Enviar actualizaciones al LCD (modo Afinador / Efectos)
    """

    def __init__(self, port: str = PORT, baud: int = BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.listeners = []   # callbacks que reciben texto del ESP32
        self.running = False

    # ------------------------------------------------------------
    # CONECTAR
    # ------------------------------------------------------------
    def conectar(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(1)
            print(f"ESP32 conectado en {self.port}")
        except Exception as e:
            print(f"❌ Error al conectar con ESP32: {e}")
            self.ser = None

    # ------------------------------------------------------------
    # ENVIAR TEXTO AL ESP32
    # ------------------------------------------------------------
    def enviar(self, mensaje: str):
        """
        Enviar texto plano al ESP32.
        """
        try:
            if self.ser:
                self.ser.write((mensaje + "\n").encode("utf-8"))
        except Exception as e:
            print(f"❌ Error enviando mensaje: {e}")

    # ------------------------------------------------------------
    # ENVIAR MENSAJE PARA LCD
    # ------------------------------------------------------------
    def enviar_lcd(self, tipo: str, l1="", l2="", l3="", l4=""):
        """
        Enviar datos al LCD del ESP32:
        Formato → tipo,l1,l2,l3,l4\n
        """
        if self.ser is None:
            return

        try:
            l1 = l1[:20]
            l2 = l2[:20]
            l3 = l3[:20]
            l4 = l4[:20]

            msg = f"{tipo},{l1},{l2},{l3},{l4}\n"
            self.ser.write(msg.encode("utf-8"))

        except Exception as e:
            print(f"❌ Error enviando LCD: {e}")

    # ------------------------------------------------------------
    # SUSCRIPTORES
    # ------------------------------------------------------------
    def agregar_listener(self, callback):
        """
        Añade un callback que recibe líneas del ESP32.
        """
        self.listeners.append(callback)

    # ------------------------------------------------------------
    # INICIAR ESCUCHA EN HILO
    # ------------------------------------------------------------
    def iniciar_escucha(self):
        if self.ser is None:
            print("⚠️ No se puede escuchar: ESP32 no conectado")
            return

        self.running = True

        def loop():
            while self.running:
                try:
                    if self.ser.in_waiting:
                        data = self.ser.readline().decode("utf-8", errors="ignore").strip()

                        for cb in self.listeners:
                            cb(data)

                except Exception as e:
                    print(f"❌ Error escuchando ESP32: {e}")
                    time.sleep(1)

                time.sleep(0.05)

        threading.Thread(target=loop, daemon=True).start()

    # ------------------------------------------------------------
    # DETENER ESCUCHA
    # ------------------------------------------------------------
    def detener(self):
        self.running = False
        try:
            if self.ser:
                self.ser.close()
        except:
            pass


# ============================================================
# FUNCIÓN GLOBAL (SHORTHAND)
# ============================================================

# El main.py puede hacer:
#
#   from core_esp32_serial import esp32, enviar_lcd
#
# Y usar enviar_lcd( ... ) directamente.
#
esp32 = ESP32Bridge()


def enviar_lcd(tipo, l1="", l2="", l3="", l4=""):
    """
    Shorthand global:
    Permite llamar enviar_lcd(...) sin tener que referirse al objeto.
    """
    esp32.enviar_lcd(tipo, l1, l2, l3, l4)
