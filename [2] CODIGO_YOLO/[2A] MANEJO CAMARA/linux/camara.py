import cv2
from picamera2 import Picamera2

print("[INFO] Iniciando cámara Raspberry Pi con Picamera2...")

# Crear objeto cámara
picam2 = Picamera2()

# Configurar resolución y la maxima resolucion sera (3280,2464)
config = picam2.create_preview_configuration(
    main={"size": (3280, 2464)}
)

picam2.configure(config)

# Iniciar cámara
picam2.start()

# Capturar una imagen
frame = picam2.capture_array()

if frame is not None:
    print("[INFO] Imagen capturada correctamente")

    # Picamera2 suele entregar imagen en RGB
    # OpenCV trabaja mejor en BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Toma de imagen con camara Raspberry Pi", frame_bgr)
    print("[INFO] Presiona cualquier tecla en la ventana para cerrar")
    cv2.waitKey(0)
else:
    print("[ERROR] No se pudo capturar la imagen")

# Cerrar recursos
cv2.destroyAllWindows()
picam2.stop()

print("[INFO] FIN DEL PROGRAMA ..........")
