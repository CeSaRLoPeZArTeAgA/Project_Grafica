import cv2
# abre miltiples camaras simultaneamente
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

# detectar y abrir todas las cámaras
camaras_abiertas = detectar_camaras_disponibles()

if not camaras_abiertas:
    print("No se encontraron camaras disponibles.")
else:
    while True:
        for i, cam in camaras_abiertas:
            ret, frame = cam.read()
            if ret:
                cv2.imshow(f"Camara {i} - Presionar 'q' para SALIR", frame)
        
        # salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar todas las cámaras y cerrar ventanas
    for _, cam in camaras_abiertas:
        cam.release()
    cv2.destroyAllWindows()
