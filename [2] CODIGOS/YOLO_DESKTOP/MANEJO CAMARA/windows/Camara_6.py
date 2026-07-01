import cv2
import os
import time
from datetime import datetime

# ======================= CONFIGURACION ========================
# Ruta absoluta donde esta el script
directorio_base = os.path.dirname(os.path.abspath(__file__))
CARPETA_RAIZ = os.path.join(directorio_base, "Fotos_Videos-Camara_6")
os.makedirs(CARPETA_RAIZ, exist_ok=True)
#creacion de subcarpetas
CARPETA_CAPTURAS = os.path.join(CARPETA_RAIZ, "capturas")
CARPETA_VIDEOS = os.path.join(CARPETA_RAIZ, "videos")
# Crear carpetas si no existen
os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
os.makedirs(CARPETA_VIDEOS, exist_ok=True)
# definicion de tiempos
TIEMPO_ENTRE_FOTOS = 5  # segundos entre capturas automaticas
DURACION_MAXIMA_SEGUNDOS = 60 * 2  # duracion maxima del programa (opcional)

# ============== FUNCION PARA DETECTAR CAMARAS ================
def detectar_camaras_disponibles(max_dispositivos=10):
    camaras = []
    for i in range(max_dispositivos):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"[INFO] Cámara detectada en el índice {i}")
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camaras.append((i, cam))
        else:
            cam.release()
    return camaras

# ==================== INICIALIZACION =========================
camaras_abiertas = detectar_camaras_disponibles()
video_writers = {}
fps = 20  # fotogramas por segundo para los videos
cuatrocc = cv2.VideoWriter_fourcc(*'XVID')  # codec para el video
inicio_programa = time.time()
ultimo_guardado = time.time()

# Crear grabadores de video
for i, cam in camaras_abiertas:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_video = os.path.join(CARPETA_VIDEOS, f"camara_{i}_{timestamp}.avi")
    video_writers[i] = cv2.VideoWriter(nombre_video, cuatrocc, fps, (640, 480))
    print(f"[INFO] Grabando video: {nombre_video}")

if not camaras_abiertas:
    print("[ERROR] No se encontraron camaras disponibles.")
    exit()

print("[INFO] Grabacion iniciada. Presiona 'q' para salir.")

# ========== BUCLE PRINCIPAL TOMA DE FOTOS Y VIDEO ==========
while True:
    frames_actuales = []
    for i, cam in camaras_abiertas:
        ret, frame = cam.read()
        if ret:
            # mostrar en ventana
            cv2.imshow(f"Camara {i} - Presiona 'q' para salir", frame)
            frames_actuales.append((i, frame))
            # grabar en video
            video_writers[i].write(frame)

    tiempo_actual = time.time()

    # Guardar imagenes cada cierto tiempo
    if tiempo_actual - ultimo_guardado >= TIEMPO_ENTRE_FOTOS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, frame in frames_actuales:
            nombre_img = os.path.join(CARPETA_CAPTURAS, f"camara_{i}_{timestamp}.png")
            cv2.imwrite(nombre_img, frame)
            print(f"[✔] Imagen guardada: {nombre_img}")
        ultimo_guardado = tiempo_actual

    # Salir con 'q' o tiempo maximo alcanzado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if tiempo_actual - inicio_programa > DURACION_MAXIMA_SEGUNDOS:
        print("[INFO] Tiempo máximo alcanzado. Finalizando...")
        break

# =================== CIERRE CAMARAS =======================
for i, cam in camaras_abiertas:
    cam.release()
    video_writers[i].release()

cv2.destroyAllWindows()

print("[INFO] Cámaras cerradas. Archivos guardados correctamente.")
print("[INFO] Cámaras cerradas. Archivos guardados en la carpeta 'RESULTADOS'........")
