import serial
import time
import threading

# ============================
# CONFIG SERIAL
# ============================
PORT = "/dev/ttyUSB0"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(1)

print("ESP32 conectado")

modo_actual = None


# ============================
# FUNCIONES PARA ENVIAR A LCD
# ============================
def enviar_lcd(tipo, l1="", l2="", l3="", l4=""):
    """
    tipo = 'A' para afinador
    tipo = 'E' para efectos
    """
    mensaje = f"{tipo},{l1},{l2},{l3},{l4}\n"
    ser.write(mensaje.encode("utf-8"))


# ============================
# LÓGICA DEL AFINADOR
# ============================
def loop_afinador():
    print("Modo Afinador activo...")

    # AQUÍ VA TODO TU CÓDIGO PESADO (FFT, mic, reconocimiento, patrones, etc.)
    # -------------------------------------------------
    # === RELLENAR CÓDIGO AQUÍ ===
    # -------------------------------------------------

    # Ejemplo temporal:
    while modo_actual == "AFINADOR":
        enviar_lcd("A", "Codigo afinador aqui", "", "", "Fila 4 = Volver")
        time.sleep(0.5)


# ============================
# LÓGICA DEL PROCESADOR DE EFECTOS
# ============================
def loop_efectos():
    print("Modo Efectos activo...")

    # === RELLENAR CÓDIGO AQUÍ ===
    # (DSP, reververancia, distorsión, etc.)
    
    # Ejemplo temporal:
    while modo_actual == "EFECTOS":
        enviar_lcd("E","Codigo efectos aqui", "", "", "Fila 4 = Volver")
        time.sleep(0.5)


# ============================
# LECTURA SERIAL DEL ESP32
# ============================
def escuchar_esp32():
    global modo_actual

    while True:
        if ser.in_waiting:
            data = ser.readline().decode("utf-8").strip()

            if data == "MODO_AFINADOR":
                modo_actual = "AFINADOR"
                threading.Thread(target=loop_afinador, daemon=True).start()

            elif data == "MODO_EFECTOS":
                modo_actual = "EFECTOS"
                threading.Thread(target=loop_efectos, daemon=True).start()


# ============================
# INICIO
# ============================
threading.Thread(target=escuchar_esp32, daemon=True).start()

print("Esperando acciones del usuario en ESP32...")

# Mantener vivo el programa
while True:
    time.sleep(1)
