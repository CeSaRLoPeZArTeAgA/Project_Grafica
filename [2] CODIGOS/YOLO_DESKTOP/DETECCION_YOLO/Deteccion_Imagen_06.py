from ultralytics import YOLO
import os
import cv2

# cargar el modelo YOLO 
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo11n.pt")
model = YOLO(model_path)

# función para detectar cámaras disponibles 
def detectar_camaras(max_test=10):
    camaras_disponibles = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            camaras_disponibles.append(i)
            cap.release()
    return camaras_disponibles

# detectar todas las cámaras 
ids_camaras = detectar_camaras()
print(f"✅ Cámaras encontradas: {ids_camaras}")

# abrir todas las camaras 
capturas = [cv2.VideoCapture(i) for i in ids_camaras]

#  ciclo principal 
while True:
    for i, cap in enumerate(capturas):
        ret, frame = cap.read()
        if not ret:
            continue

        # inferencia YOLO
        results = model(frame)
        annotated = results[0].plot()

        # mostrar en ventana separada por cámara
        cv2.imshow(f"Cámara {i}", annotated)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar recursos 
for cap in capturas:
    cap.release()
cv2.destroyAllWindows()
