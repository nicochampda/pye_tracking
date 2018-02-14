import numpy as np
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

    return frame

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    detection = detect_face(frame)

    # Display the resulting frame
    cv2.imshow('frame', detection)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


