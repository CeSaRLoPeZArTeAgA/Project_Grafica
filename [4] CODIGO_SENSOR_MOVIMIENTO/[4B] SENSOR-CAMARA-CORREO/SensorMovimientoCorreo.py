import os
import time
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import RPi.GPIO as GPIO


# ==========================
# CONFIGURACIÓN GPIO
# ==========================

PIN_SENSOR = 23  # GPIO 23 = pin físico 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


# ==========================
# CONFIGURACIÓN CORREO
# ==========================
def leer_variable(nombre):
    """
    Primero intenta leer la variable desde el entorno.
    Si no existe, intenta leerla desde ~/.bashrc.
    """
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


def enviar_correo(fecha_hora):
    """
    Envía un correo notificando que se detectó movimiento.
    """

    if EMAIL_USER is None or EMAIL_PASSWORD is None or EMAIL_TO is None:
        print("[ERROR] Faltan variables de entorno del correo")
        print("[INFO] Debes configurar GMAIL_USER, GMAIL_APP_PASSWORD y GMAIL_TO")
        return

    asunto = "Alerta de movimiento detectado"

    cuerpo = f"""
Se detectó movimiento en el sensor conectado al GPIO 23.

Fecha y hora de detección:
{fecha_hora}

Dispositivo:
Raspberry Pi
"""

    mensaje = MIMEMultipart()
    mensaje["From"] = EMAIL_USER
    mensaje["To"] = EMAIL_TO
    mensaje["Subject"] = asunto

    mensaje.attach(MIMEText(cuerpo, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(mensaje)

        print("[INFO] Correo enviado correctamente")

    except Exception as e:
        print("[ERROR] No se pudo enviar el correo")
        print(e)


print("[INFO] Sistema iniciado")
print("[INFO] Esperando movimiento en GPIO 23...")
print("[INFO] Presiona CTRL + C para salir")


# ==========================
# BUCLE PRINCIPAL
# ==========================

movimiento_anterior = GPIO.LOW
TIEMPO_ESPERA = 10  # segundos para evitar muchos correos seguidos

try:
    while True:
        movimiento_actual = GPIO.input(PIN_SENSOR)

        # Detecta solo el cambio de NO movimiento a movimiento
        if movimiento_actual == GPIO.HIGH and movimiento_anterior == GPIO.LOW:
            fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            print(f"[MOVIMIENTO] Movimiento detectado: {fecha_hora}")

            enviar_correo(fecha_hora)

            # Evita enviar muchos correos seguidos
            time.sleep(TIEMPO_ESPERA)

        movimiento_anterior = movimiento_actual
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Programa detenido por el usuario")

finally:
    GPIO.cleanup()
    print("[INFO] GPIO liberado correctamente")