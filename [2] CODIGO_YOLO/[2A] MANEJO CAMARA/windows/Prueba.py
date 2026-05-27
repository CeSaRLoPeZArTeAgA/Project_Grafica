'''PROGRAMA QUE LEE UNA IMAGEN Y LA MUESTRA'''
import os
import cv2

# ruta absoluta a la imagen dentro de la carpeta del script
script_dir = os.path.dirname(os.path.abspath(__file__))
imagen_path = os.path.join(script_dir, 'prueba.jpg')

imagen = cv2.imread(imagen_path)
if imagen is None:
    raise FileNotFoundError(f'No se pudo abrir la imagen: {imagen_path}')

cv2.imshow('IMAGEN DE PRUEBA', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()