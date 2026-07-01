import cv2
from picamera2 import Picamera2

print("[INFO] Iniciando video con Picamera2...")

picam2 = Picamera2()

# Configurar resolución y la maxima resolucion sera (3280,2464)
config = picam2.create_preview_configuration(
    main={"size": (3280, 2464)}
)

picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Video Raspberry Pi", frame_bgr)

    # Presiona q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()

print("[INFO] FIN DEL PROGRAMA")