import cv2

#detecta que camaras estan habilitadas y solo abre la camara
#que queremos usar

print("[INFO] Buscando cámaras disponibles...")
camaras_encontradas = []

# vemos que camaras estan habilitadas en el computador
for i in range(5):  # intenta con los primeros 5 dispositivos
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[INFO] Camara encontrada en el índice {i}")
        cap.release()
        
# Abrir la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[INFO] No se pudo abrir la cámara")
else:
    ret, frame = cap.read()
    if ret:
        # mostrar el "TEXTO" , "IMAGEN(frame)" en una ventana
        cv2.imshow("Toma de una imagen con la camara de la computadora", frame)
        # Espera indefinida hasta que se presione una tecla
        cv2.waitKey(0)
        
# liberar la cámara y cerrar ventanas
cap.release()
# cierra todas las ventanas
cv2.destroyAllWindows()
print("[INFO] FIN DEL PROGRAMA ..........")
