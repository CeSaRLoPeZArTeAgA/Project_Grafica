import os
import time
import smtplib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage

import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2

from DepthAnythingONNX_RPi import DepthAnythingONNX, colorizar_profundidad


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

PIN_SENSOR = 23                  # GPIO 23 = pin físico 16

# Se conserva el tamaño usado en SensorMovimientoFotoCorreo.py.
ANCHO = 3280
ALTO = 2464

# Valor de respaldo. Si el ONNX tiene tamaño fijo, la clase lo detecta automáticamente.
INPUT_SIZE_DEPTH = 518
NUM_THREADS = 2
TIEMPO_ESPERA = 10               # segundos para evitar muchos correos seguidos

BASE_DIR = Path(__file__).resolve().parent
MODELO_DEPTH = BASE_DIR / "models" / "depth_anything_v2_vits.onnx"
CARPETA_FOTOS = BASE_DIR / "capturas_movimiento_depth"
CARPETA_FOTOS.mkdir(exist_ok=True)


# ============================================================
# CONFIGURACIÓN DEL CORREO
# ============================================================

def leer_variable(nombre):
    valor = os.getenv(nombre)

    if valor:
        return valor

    ruta_bashrc = os.path.expanduser("~/.bashrc")

    if os.path.exists(ruta_bashrc):
        with open(ruta_bashrc, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.strip()
                patron = f"export {nombre}="

                if linea.startswith(patron):
                    valor = linea.split("=", 1)[1].strip()
                    valor = valor.strip('"')
                    valor = valor.strip("'")
                    return valor

    return None


EMAIL_USER = leer_variable("GMAIL_USER")
EMAIL_PASSWORD = leer_variable("GMAIL_APP_PASSWORD")
EMAIL_TO = leer_variable("GMAIL_TO")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


# ============================================================
# FUNCIONES DE IMAGEN Y CORREO
# ============================================================

def escribir_texto(frame_bgr, texto, escala=2.2, grosor=5):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    x = 80
    y = frame_bgr.shape[0] - 100

    (text_w, text_h), baseline = cv2.getTextSize(texto, fuente, escala, grosor)

    cv2.rectangle(
        frame_bgr,
        (x - 30, y - text_h - 30),
        (x + text_w + 30, y + baseline + 30),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame_bgr,
        texto,
        (x, y),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA,
    )

    return frame_bgr


def tomar_foto_y_profundidad(picam2, depth_model, fecha_hora):
    ahora = datetime.now()
    base_nombre = ahora.strftime("movimiento_depth_%Y%m%d_%H%M%S")

    ruta_foto = CARPETA_FOTOS / f"{base_nombre}_foto.jpg"
    ruta_depth = CARPETA_FOTOS / f"{base_nombre}_depth.jpg"
    ruta_comparacion = CARPETA_FOTOS / f"{base_nombre}_comparacion.jpg"

    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    texto = f"Movimiento detectado: {fecha_hora}"
    frame_anotado = escribir_texto(frame_bgr.copy(), texto)

    depth_norm = depth_model.infer(frame_bgr)
    depth_color = colorizar_profundidad(depth_norm)

    cv2.imwrite(str(ruta_foto), frame_anotado, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(ruta_depth), depth_color, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Imagen comparativa en tamaño manejable para correo.
    foto_small = cv2.resize(frame_anotado, (640, 480), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_color, (640, 480), interpolation=cv2.INTER_AREA)
    comparacion = cv2.hconcat([foto_small, depth_small])

    cv2.putText(
        comparacion,
        "Foto original",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparacion,
        "Mapa de profundidad relativa",
        (670, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(ruta_comparacion), comparacion, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return ruta_foto, ruta_depth, ruta_comparacion


def enviar_correo_con_archivos(fecha_hora, rutas):
    if EMAIL_USER is None or EMAIL_PASSWORD is None or EMAIL_TO is None:
        print("[ERROR] Faltan variables de entorno del correo")
        print("[INFO] Configura GMAIL_USER, GMAIL_APP_PASSWORD y GMAIL_TO")
        return

    mensaje = EmailMessage()
    mensaje["From"] = EMAIL_USER
    mensaje["To"] = EMAIL_TO
    mensaje["Subject"] = "Alerta de movimiento con estimación de profundidad"

    cuerpo = f"""
Se detectó movimiento en el sensor conectado al GPIO 23.

Fecha y hora:
{fecha_hora}

Se adjuntan:
1. Fotografía original del evento.
2. Mapa de profundidad relativa estimado con Depth Anything V2.
3. Comparación visual entre foto y profundidad.

Dispositivo:
Raspberry Pi 4B
"""

    mensaje.set_content(cuerpo)

    for ruta in rutas:
        with open(ruta, "rb") as archivo:
            datos = archivo.read()

        mensaje.add_attachment(
            datos,
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(ruta),
        )

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(mensaje)

        print("[INFO] Correo enviado correctamente")

    except Exception as e:
        print("[ERROR] No se pudo enviar el correo")
        print(e)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("[INFO] Sistema iniciado: sensor + foto + Depth Anything + correo")
    print(f"[INFO] Modelo Depth: {MODELO_DEPTH}")

    if not MODELO_DEPTH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODELO_DEPTH}\n"
            "Coloca depth_anything_v2_vits.onnx dentro de la carpeta models."
        )

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    picam2 = Picamera2()

    try:
        config = picam2.create_still_configuration(
            main={
                "size": (ANCHO, ALTO),
                "format": "RGB888",
            }
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        depth_model = DepthAnythingONNX(
            MODELO_DEPTH,
            input_size=INPUT_SIZE_DEPTH,
            num_threads=NUM_THREADS,
        )

        print("[INFO] Esperando movimiento en GPIO 23...")
        print("[INFO] Presiona CTRL + C para salir")

        movimiento_anterior = GPIO.LOW

        while True:
            movimiento_actual = GPIO.input(PIN_SENSOR)

            if movimiento_actual == GPIO.HIGH and movimiento_anterior == GPIO.LOW:
                fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print(f"[MOVIMIENTO] Movimiento detectado: {fecha_hora}")

                ruta_foto, ruta_depth, ruta_comparacion = tomar_foto_y_profundidad(
                    picam2,
                    depth_model,
                    fecha_hora,
                )

                print(f"[INFO] Foto guardada en: {ruta_foto}")
                print(f"[INFO] Profundidad guardada en: {ruta_depth}")
                print(f"[INFO] Comparación guardada en: {ruta_comparacion}")

                enviar_correo_con_archivos(
                    fecha_hora,
                    [ruta_foto, ruta_depth, ruta_comparacion],
                )

                time.sleep(TIEMPO_ESPERA)

            movimiento_anterior = movimiento_actual
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Programa detenido por el usuario")

    finally:
        picam2.stop()
        GPIO.cleanup()
        print("[INFO] Cámara y GPIO liberados correctamente")


if __name__ == "__main__":
    main()
