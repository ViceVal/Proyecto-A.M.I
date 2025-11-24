# ===============================================================
# main.py
# Punto de entrada principal del sistema completo
# ===============================================================

import tkinter as tk
from gui_interfaz import AfinadorGuitarraAcusticaGUI
from core_esp32_serial import ESP32Bridge
import time


def main():

    # -----------------------------------------------------------
    # 1. Crear puente con ESP32
    # -----------------------------------------------------------
    esp32 = ESP32Bridge()
    esp32.conectar()

    # -----------------------------------------------------------
    # 2. Crear la ventana principal
    # -----------------------------------------------------------
    root = tk.Tk()
    app = AfinadorGuitarraAcusticaGUI(root)

    # -----------------------------------------------------------
    # 3. Enlazar datos desde ESP32 → GUI
    # -----------------------------------------------------------
    if esp32.ser is not None:
        esp32.agregar_listener(app.on_esp32_data)
        esp32.iniciar_escucha()
        print("Escuchando mensajes del ESP32...")
    else:
        print("⚠ No se pudo conectar al ESP32. Continuando sin conexión serial.")

    # -----------------------------------------------------------
    # 4. Manejar cierre limpio
    # -----------------------------------------------------------
    def on_closing():
        print("Cerrando sistema...")
        try:
            app.cerrar_aplicacion()
        except:
            pass
        try:
            if esp32.ser:
                esp32.ser.close()
        except:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # -----------------------------------------------------------
    # 5. Iniciar GUI
    # -----------------------------------------------------------
    root.mainloop()


# Ejecutar MAIN
if __name__ == "__main__":
    main()
