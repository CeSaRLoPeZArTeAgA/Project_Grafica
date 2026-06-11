from ultralytics import YOLO
import os
# deteccion de objetos, en imagen, uno en uno
# se usa los parametros del propio modelo para identificar

# Obtener el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta local al modelo YOLO
model_path = os.path.join(script_dir, "yolo11n.pt")

# cargar modelo YOLO 11 nano pt(deteccion)
model = YOLO(model_path)

# path de la imagen donde se aplicara la deteccion, solo una imagen a la vez
print("Ruta: ",script_dir)
source = os.path.join(script_dir, "Imagen_Deteccion", "persona_02.jpg")
#source="./Imagen_Deteccion/persona_02.jpg"

# realizamos la prediccion con YOLO de la imagen
result=model(source)
print(result) # se visualiza x terminal

# visualizamos el resultado, solo una imagen
result[0].show()
