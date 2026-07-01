import cv2

# abre dos camaras simultaneamente, se pone explicitamente

# Abrir las dos cámaras (0 y 1)
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

# Verificar si ambas cámaras se abrieron correctamente
if not cam0.isOpened():
    print("No se pudo abrir la camara 0")
if not cam1.isOpened():
    print("No se pudo abrir la camara 1")

if cam0.isOpened() and cam1.isOpened():
    while True:
        ret0, frame0 = cam0.read()
        ret1, frame1 = cam1.read()

        if ret0:
            cv2.imshow("Camara 0 - Presione q para salir", frame0)
        if ret1:
            cv2.imshow("Camara 1 - Presione q para salir", frame1)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cam0.release()
    cam1.release()
    cv2.destroyAllWindows()
else:
    print("Una o ambas cámaras no se pudieron abrir correctamente.")
