import os
import time
from datetime import datetime

import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ==========================
# CONFIGURACIÓN
# ==========================

PIN_SENSOR = 23          # GPIO 23, pin físico 16
ANCHO = 3280             # Máxima resolución IMX219
ALTO = 2464
CARPETA = "capturas_movimiento"
TIEMPO_ESPERA = 5        # segundos para evitar muchas fotos seguidas

# Crear carpeta para guardar fotos
os.makedirs(CARPETA, exist_ok=True)

# ==========================
# CONFIGURAR GPIO
# ==========================

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# ==========================
# CONFIGURAR CÁMARA
# ==========================

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={
        "size": (ANCHO, ALTO),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

# Espera para estabilizar cámara
time.sleep(2)

print("[INFO] Sistema iniciado")
print("[INFO] Esperando movimiento en GPIO 23...")
print("[INFO] Presiona CTRL + C para salir")


def tomar_foto_con_fecha():
    # Fecha y hora actual
    ahora = datetime.now()
    fecha_texto = ahora.strftime("%d/%m/%Y %H:%M:%S")
    nombre_archivo = ahora.strftime("movimiento_%Y%m%d_%H%M%S.jpg")
    ruta_archivo = os.path.join(CARPETA, nombre_archivo)

    # Capturar imagen
    frame_rgb = picam2.capture_array()

    # Convertir de RGB a BGR para OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Texto que irá en la imagen
    texto = f"Movimiento detectado: {fecha_texto}"

    # Parámetros del texto
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 2.2
    grosor = 5

    # Posición del texto
    x = 80
    y = ALTO - 100

    # Calcular tamaño del texto
    (text_w, text_h), baseline = cv2.getTextSize(texto, fuente, escala, grosor)

    # Fondo negro para que el texto se lea bien
    cv2.rectangle(
        frame_bgr,
        (x - 30, y - text_h - 30),
        (x + text_w + 30, y + baseline + 30),
        (0, 0, 0),
        -1
    )

    # Escribir texto en color blanco
    cv2.putText(
        frame_bgr,
        texto,
        (x, y),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA
    )

    # Guardar imagen
    cv2.imwrite(ruta_archivo, frame_bgr)

    print(f"[MOVIMIENTO] {fecha_texto}")
    print(f"[INFO] Foto guardada: {ruta_archivo}")


try:
    movimiento_anterior = GPIO.LOW

    while True:
        movimiento_actual = GPIO.input(PIN_SENSOR)

        # Detecta solo el cambio de NO movimiento a movimiento
        if movimiento_actual == GPIO.HIGH and movimiento_anterior == GPIO.LOW:
            tomar_foto_con_fecha()
            time.sleep(TIEMPO_ESPERA)

        movimiento_anterior = movimiento_actual
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Programa detenido por el usuario")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("[INFO] Cámara y GPIO liberados correctamente")