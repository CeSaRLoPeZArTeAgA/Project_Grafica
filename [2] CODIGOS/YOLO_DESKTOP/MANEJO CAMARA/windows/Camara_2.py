import cv2
# Abrir la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame")
        break

    # mostrar el texto y frame en una ventana
    cv2.imshow("Camara Computadora - Presiona q para Salir", frame)

    # Salir si el usuario presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
