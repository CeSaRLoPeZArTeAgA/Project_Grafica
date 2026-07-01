from ultralytics import YOLO
from picamera2 import Picamera2
import os
import cv2
import time

# ============================================================
# DETECCIÓN DE OBJETOS EN TIEMPO REAL CON YOLO-NCNN Y PICAMERA2
# Raspberry Pi 4B - Cámara CSI IMX219
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

script_dir = os.path.dirname(os.path.abspath(__file__))

# Usar modelo NCNN, no el .pt
model_path = os.path.join(script_dir, "yolo11n_ncnn_model")

if not os.path.exists(model_path):
    print(f"[ERROR] No se encontro el modelo NCNN en: {model_path}")
    print("[INFO] Debes copiar la carpeta yolo11n_ncnn_model a esta ruta.")
    exit()

print("[INFO] Cargando modelo YOLO NCNN...")
model = YOLO(model_path)
print("[INFO] Modelo NCNN cargado correctamente.")

print("[INFO] Iniciando camara con Picamera2...")
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()
time.sleep(1)

print("[INFO] Camara iniciada correctamente.")
print("[INFO] Presiona 'q' para salir.")

while True:
    frame_rgb = picam2.capture_array()

    if frame_rgb is None:
        print("[ERROR] No se pudo capturar frame.")
        break

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=frame_bgr,
        imgsz=320,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO NCNN - Raspberry Pi 4B", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()

print("[INFO] Programa finalizado correctamente.")