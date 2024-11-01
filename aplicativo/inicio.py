import cv2 #pip install opencv-python
#importando mediapipe
import mediapipe as mp #pip install mediapipe

# capturar a camêra
cap = cv2.VideoCapture(1)



# enquanto a camera estiver aberta
while cap.isOpened():
    # sucesso é booleana-0 e 1
    sucesso,frame = cap.read()
    if not sucesso:
        print('ignorando o frame vazio da camera')
        continue
    cv2.imshow('Camera',frame)

    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
#fecha a captura
cap.release()
cv2.destroyAllWindows()