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

# Cargamos la imagen de entrada de donde lee openCV
image_path = os.path.join(script_dir, "Imagen_Deteccion", "persona_01.jpg")
image = cv2.imread(image_path)
#image = cv2.imread("./MODELO_YOLO/Imagen_Deteccion/persona_01.jpg")

# Realizamos la inferencia sobre la imagen
results = model(image)

for result in results:
    # Extraemos los nombres de las clases detectadas
    names = [result.names[int(label)] for label in result.boxes.cls]
    # Extraemos las coordenadas de las cajas delimitadoras
    xyxys = result.boxes.xyxy
    for i, xyxy in enumerate(xyxys):
        x1, y1 = int(xyxy[0]), int(xyxy[1])
        x2, y2 = int(xyxy[2]), int(xyxy[3])
        # Dibujar la caja
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # Poner la etiqueta
        cv2.putText(image, names[i], (x1, y1 - 5), 1, 1.1, (0, 0, 255), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow("Deteccion de Objetos [3] - Presiona 'q' para Salir", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()