from ultralytics import YOLO
import os
# deteccion de objetos, en imagen, cuatro a la vez, desde un directorio dado
# se usa los parametros del propio modelo para identificar

# Obtener el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta local al modelo YOLO
model_path = os.path.join(script_dir, "yolo11n.pt")

# cargar modelo YOLO 11 nano pt(deteccion)
model = YOLO(model_path)

# Lista de nombres de imágenes
image_names = ["persona_01.jpg", "persona_02.jpg", "persona_03.jpg", "persona_04.jpg"]

# Por esta (porque Imagen_Deteccion está dentro de MODELO_YOLO):
image_folder = os.path.join(script_dir, "Imagen_Deteccion")
image_paths = [os.path.join(image_folder, name) for name in image_names]

# Normalizar rutas (opcional pero útil)
image_paths = [os.path.normpath(path) for path in image_paths]

# Mostrar rutas para verificar
for path in image_paths:
    print("Imagen a procesar:", path)

# Realizar la detección
results = model(image_paths)

# visualizamos el resultado, todas las imagenes
for result in results:
    print("===============================================================================")
    #print(result)
    print(result.boxes)
    result.show()