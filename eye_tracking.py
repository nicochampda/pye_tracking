import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_face(frame):

    global face_cascade
    global eye_cascade

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extraction des faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Pour toutes les faces detectees
    for (x,y,w,h) in faces:
        # Dessine un rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

        # Regions d'interet
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Extraction des yeux
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #Region d'interet autour des yeux
            roi_eye_gray = roi_gray[ey:ey+eh][ex:ex+ew]
            roi_eye_color = roi_color[ey:ey+eh][ex:ex+ew]
            if np.min(roi_eye_gray.shape) > 5:
                c = detect_eye_center(roi_eye_gray)
                cv2.line(roi_eye_color, tuple(c-np.array((0,4))), tuple(c+np.array((0,4))), (255,255,0), 1)
                cv2.line(roi_eye_color, tuple(c-np.array((4,0))), tuple(c+np.array((4,0))), (255,255,0),1)
                
    return frame

def detect_eye_center(img):
    print(img.shape)
    grad_x = np.gradient(img, axis=0)
    grad_y = np.gradient(img, axis=1)
    x,y = img.shape
    c = 0,0
    max_val = 0
    for i in range(x):
        for j in range(y):
            somme = 0
            for k in range(x):
                for l in range(y):
                    if (i,j) != (k,l):
                        dx = k - i
                        dy = l - j
                        norm = np.sqrt(dx**2 + dy**2)
                        somme += (grad_x[k][l]*(dx/norm) + grad_y[k][l]*(dy/norm))**2
                    
            if somme > max_val:
                max_val = somme
                c = i,j
    return c
    
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cap.release()
    detection = detect_face(frame)

    # Display the resulting frame
    cv2.imshow('frame', detection)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


