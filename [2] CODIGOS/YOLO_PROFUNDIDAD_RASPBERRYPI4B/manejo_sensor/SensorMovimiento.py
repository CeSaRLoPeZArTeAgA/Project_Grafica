import RPi.GPIO as GPIO
import time
from datetime import datetime

PIN_SENSOR = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

print("[INFO] Esperando movimiento en GPIO 23...")
print("[INFO] Presiona CTRL + C para salir")

movimiento_anterior = GPIO.LOW

try:
    while True:
        movimiento_actual = GPIO.input(PIN_SENSOR)

        # Detecta solo cuando cambia de NO movimiento a movimiento
        if movimiento_actual == GPIO.HIGH and movimiento_anterior == GPIO.LOW:
            fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"Movimiento detectado: {fecha_hora}")

        movimiento_anterior = movimiento_actual
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Programa detenido")

finally:
    GPIO.cleanup()