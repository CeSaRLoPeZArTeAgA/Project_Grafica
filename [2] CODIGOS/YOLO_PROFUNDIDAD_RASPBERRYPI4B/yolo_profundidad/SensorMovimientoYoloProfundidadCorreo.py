import os
import time
import smtplib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage

import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

from DepthAnythingONNX_RPi import DepthAnythingONNX, colorizar_profundidad, valor_roi_profundidad


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

PIN_SENSOR = 23

# Para inferencia combinada se usa resolución moderada por rendimiento.
ANCHO = 640
ALTO = 480

# Valor de respaldo. Si el ONNX tiene tamaño fijo, la clase lo detecta automáticamente.
INPUT_SIZE_DEPTH = 518
YOLO_IMGSZ = 320
CONF_YOLO = 0.35

# Umbral de profundidad relativa. Ajustar experimentalmente.
UMBRAL_CERCANIA = 0.62
NEAR_HIGH = True              # True: valores altos indican mayor cercanía

NUM_THREADS = 2
TIEMPO_ESPERA = 20            # segundos para evitar correos repetidos

BASE_DIR = Path(__file__).resolve().parent
MODELO_YOLO = BASE_DIR / "models" / "yolo11n_ncnn_model"
MODELO_DEPTH = BASE_DIR / "models" / "depth_anything_v2_vits.onnx"
CARPETA_ALERTAS = BASE_DIR / "alertas_yolo_depth"
CARPETA_ALERTAS.mkdir(exist_ok=True)


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
# CORREO
# ============================================================

def enviar_correo_alerta(fecha_hora, ruta_imagen, cantidad_personas):
    if EMAIL_USER is None or EMAIL_PASSWORD is None or EMAIL_TO is None:
        print("[ERROR] Faltan variables de entorno del correo")
        print("[INFO] Configura GMAIL_USER, GMAIL_APP_PASSWORD y GMAIL_TO")
        return

    mensaje = EmailMessage()
    mensaje["From"] = EMAIL_USER
    mensaje["To"] = EMAIL_TO
    mensaje["Subject"] = "Alerta: persona cercana detectada"

    cuerpo = f"""
El sistema detectó una persona dentro del umbral de cercanía.

Fecha y hora:
{fecha_hora}

Cantidad de personas detectadas:
{cantidad_personas}

Criterio de activación:
Sensor PIR + YOLO(person) + Depth Anything V2.

Dispositivo:
Raspberry Pi 4B
"""

    mensaje.set_content(cuerpo)

    with open(ruta_imagen, "rb") as archivo:
        datos = archivo.read()

    mensaje.add_attachment(
        datos,
        maintype="image",
        subtype="jpeg",
        filename=os.path.basename(ruta_imagen),
    )

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(mensaje)
        print("[INFO] Correo de alerta enviado correctamente")
    except Exception as e:
        print("[ERROR] No se pudo enviar el correo")
        print(e)


# ============================================================
# PROCESAMIENTO
# ============================================================

def es_persona_cercana(valor_depth):
    if valor_depth is None:
        return False

    if NEAR_HIGH:
        return valor_depth >= UMBRAL_CERCANIA

    return valor_depth <= UMBRAL_CERCANIA


def procesar_frame(frame_bgr, yolo_model, depth_model):
    results = yolo_model.predict(
        frame_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONF_YOLO,
        classes=[0],       # person en COCO
        verbose=False,
    )

    depth_norm = depth_model.infer(frame_bgr)
    depth_color = colorizar_profundidad(depth_norm)

    alerta = False
    cantidad_personas = 0
    frame_anotado = frame_bgr.copy()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            cantidad_personas += 1

            z_rel = valor_roi_profundidad(
                depth_norm,
                x1,
                y1,
                x2,
                y2,
                margen=0.15,
            )

            cercana = es_persona_cercana(z_rel)
            alerta = alerta or cercana

            color = (0, 0, 255) if cercana else (0, 255, 0)
            if z_rel is None:
                etiqueta = f"persona {conf:.2f}"
            else:
                etiqueta = f"persona {conf:.2f} | z={z_rel:.2f}"

            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_anotado,
                etiqueta,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    depth_small = cv2.resize(
        depth_color,
        (frame_anotado.shape[1], frame_anotado.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    comparacion = cv2.hconcat([frame_anotado, depth_small])

    cv2.putText(
        comparacion,
        "YOLO + profundidad relativa",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparacion,
        "Depth Anything V2",
        (frame_anotado.shape[1] + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return alerta, cantidad_personas, comparacion


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("[INFO] Sistema iniciado: PIR + YOLO + Depth Anything + correo")

    if not MODELO_YOLO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo YOLO NCNN: {MODELO_YOLO}\n"
            "Coloca la carpeta yolo11n_ncnn_model dentro de models/."
        )

    if not MODELO_DEPTH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo Depth ONNX: {MODELO_DEPTH}\n"
            "Coloca depth_anything_v2_vits.onnx dentro de models/."
        )

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    picam2 = Picamera2()

    try:
        config = picam2.create_preview_configuration(
            main={
                "size": (ANCHO, ALTO),
                "format": "RGB888",
            }
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        print("[INFO] Cargando YOLO NCNN...")
        yolo_model = YOLO(str(MODELO_YOLO))

        print("[INFO] Cargando Depth Anything ONNX...")
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

                frame_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                alerta, cantidad_personas, comparacion = procesar_frame(
                    frame_bgr,
                    yolo_model,
                    depth_model,
                )

                nombre = datetime.now().strftime("evento_%Y%m%d_%H%M%S.jpg")
                ruta_evento = CARPETA_ALERTAS / nombre
                cv2.imwrite(str(ruta_evento), comparacion, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"[INFO] Imagen del evento guardada: {ruta_evento}")

                if alerta:
                    print("[ALERTA] Persona cercana detectada")
                    enviar_correo_alerta(fecha_hora, ruta_evento, cantidad_personas)
                    time.sleep(TIEMPO_ESPERA)
                else:
                    print("[INFO] Movimiento detectado, pero no se activó alerta de cercanía")

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
