from ultralytics import YOLO
import os
import cv2

# deteccion de objetos, con open CV, una imagen a la  vez, desde un directorio dado
# se usa los parametros del propio modelo para identificar

# Obtener el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta local al modelo YOLO
model_path = os.path.join(script_dir, "yolo11n.pt")

# cargar modelo YOLO 11 nano pt(deteccion)
model = YOLO(model_path)


#cargamos el video del webcam
cap=cv2.VideoCapture(0)

while cap.isOpened():
    # leemos el frame del video
    ret, frame=cap.read()
    if not ret:
        break

    # realizamos la inferencia de YOLO sobre el frame
    results=model(frame)

    # extraemos los resultados
    annotated_frame=results[0].plot()
    #print(annotated_frame)

    # visualizamos los resultados
    cv2.imshow("Deteccion de Objetos [5] - Presiona 'q' para Salir",annotated_frame)

    # el ciclo se rompe al presionar ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()