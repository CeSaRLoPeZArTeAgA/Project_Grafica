import os
import time
import smtplib
from datetime import datetime
from email.message import EmailMessage

import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

PIN_SENSOR = 23          # GPIO 23 = pin físico 16

ANCHO = 3280             # Máxima resolución de tu cámara IMX219
ALTO = 2464

CARPETA_FOTOS = "capturas_movimiento"
TIEMPO_ESPERA = 10       # segundos para evitar muchos correos seguidos

os.makedirs(CARPETA_FOTOS, exist_ok=True)


# ============================================================
# CONFIGURACIÓN DEL CORREO
# ============================================================

def leer_variable(nombre):
    valor = os.getenv(nombre)

    if valor:
        return valor

    ruta_bashrc = os.path.expanduser("~/.bashrc")

    if os.path.exists(ruta_bashrc):
        with open(ruta_bashrc, "r") as archivo:
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
# CONFIGURACION GPIO
# ============================================================

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


# ============================================================
# CONFIGURACION CAMARA
# ============================================================

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={
        "size": (ANCHO, ALTO),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)


# ============================================================
# FUNCION PARA TOMAR FOTO CON FECHA Y HORA
# ============================================================

def tomar_foto_con_fecha(fecha_hora):
    ahora = datetime.now()
    nombre_foto = ahora.strftime("movimiento_%Y%m%d_%H%M%S.jpg")
    ruta_foto = os.path.join(CARPETA_FOTOS, nombre_foto)

    frame_rgb = picam2.capture_array()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    texto = f"Movimiento detectado: {fecha_hora}"

    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 2.2
    grosor = 5

    x = 80
    y = ALTO - 100

    (text_w, text_h), baseline = cv2.getTextSize(
        texto,
        fuente,
        escala,
        grosor
    )

    cv2.rectangle(
        frame_bgr,
        (x - 30, y - text_h - 30),
        (x + text_w + 30, y + baseline + 30),
        (0, 0, 0),
        -1
    )

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

    cv2.imwrite(
        ruta_foto,
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    return ruta_foto


# ============================================================
# FUNCION PARA ENVIAR CORREO CON FOTO
# ============================================================

def enviar_correo_con_foto(fecha_hora, ruta_foto):
    if EMAIL_USER is None or EMAIL_PASSWORD is None or EMAIL_TO is None:
        print("[ERROR] Faltan variables de entorno del correo")
        print("[INFO] Configura GMAIL_USER, GMAIL_APP_PASSWORD y GMAIL_TO")
        return

    mensaje = EmailMessage()

    mensaje["From"] = EMAIL_USER
    mensaje["To"] = EMAIL_TO
    mensaje["Subject"] = "Alerta de movimiento detectado"

    cuerpo = f"""
Se detectó movimiento en el sensor conectado al GPIO 23.

Fecha y hora:
{fecha_hora}

La imagen capturada se adjunta en este correo.

Dispositivo:
Raspberry Pi
"""

    mensaje.set_content(cuerpo)

    with open(ruta_foto, "rb") as archivo:
        datos_imagen = archivo.read()

    nombre_archivo = os.path.basename(ruta_foto)

    mensaje.add_attachment(
        datos_imagen,
        maintype="image",
        subtype="jpeg",
        filename=nombre_archivo
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

print("[INFO] Sistema iniciado")
print("[INFO] Esperando movimiento en GPIO 23...")
print("[INFO] Presiona CTRL + C para salir")

movimiento_anterior = GPIO.LOW

try:
    while True:
        movimiento_actual = GPIO.input(PIN_SENSOR)

        if movimiento_actual == GPIO.HIGH and movimiento_anterior == GPIO.LOW:
            fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            print(f"[MOVIMIENTO] Movimiento detectado: {fecha_hora}")

            ruta_foto = tomar_foto_con_fecha(fecha_hora)

            print(f"[INFO] Foto guardada en: {ruta_foto}")

            enviar_correo_con_foto(fecha_hora, ruta_foto)

            time.sleep(TIEMPO_ESPERA)

        movimiento_anterior = movimiento_actual
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Programa detenido por el usuario")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("[INFO] Cámara y GPIO liberados correctamente")