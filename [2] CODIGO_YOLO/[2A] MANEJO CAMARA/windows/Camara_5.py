import cv2
import os
from datetime import datetime

# detecta las camaras instaladas
def detectar_camaras_disponibles(max_dispositivos=10):
    camaras = []
    for i in range(max_dispositivos):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"[INFO] Camara detectada en el indice {i}")
            camaras.append((i, cam))
        else:
            cam.release()
    return camaras

# Ruta absoluta donde está el script
directorio_base = os.path.dirname(os.path.abspath(__file__))
ruta_salida = os.path.join(directorio_base, "Solo_Fotos-Camara_5")
os.makedirs(ruta_salida, exist_ok=True)

# Detectar y abrir todas las cámaras
camaras_abiertas = detectar_camaras_disponibles()

if not camaras_abiertas:
    print("No se encontraron camaras disponibles.")
else:
    print("[INFO] Presiona 's' para capturar fotos, 'q' para salir.")
    while True:
        frames_actuales = []
        for i, cam in camaras_abiertas:
            ret, frame = cam.read()
            if ret:
                cv2.imshow(f"Camara {i} - Presiona 's' para capturar fotos, 'q' para salir", frame)
                frames_actuales.append((i, frame))
        
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, frame in frames_actuales:
                nombre_archivo = f"{ruta_salida}/camara_{i}_{timestamp}.png"
                cv2.imwrite(nombre_archivo, frame)
                print(f"[✔] Imagen guardada: {nombre_archivo}")
        
        elif tecla == ord('q'):
            break

    # Liberar cámaras y cerrar ventanas
    for _, cam in camaras_abiertas:
        cam.release()
    cv2.destroyAllWindows()
