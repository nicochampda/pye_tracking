import numpy as np
import cv2
import matplotlib.pyplot as plt
from open_image import open_image
from open_image import open_eye_pos

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_face(frame, is_gray = 'False'):

    global face_cascade
    global eye_cascade

    gray = None
    if is_gray:
        gray = np.copy(frame)
    else:
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

def main():

    cap = cv2.VideoCapture(0)
    
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

def test_dataset():
    for i in range(4):

        # Construction du nom de fichier
        img_name = str(i)
        if len(img_name) == 1:
            img_name = '000' + img_name
        elif len(img_name) == 2:
            img_name = '00' + img_name
        elif len(img_name) == 3:
            img_name = '0' + img_name

        # Ouverture du premier fichier
        img = open_image('BioID-FaceDatabase-V1.2/BioID_' + img_name + '.pgm')
        positions = open_eye_pos('BioID-FaceDatabase-V1.2/BioID_' + img_name + '.eye')

        detection = detect_face(img, is_gray = True)

        c = positions[0], positions[1]
        cv2.line(detection, tuple(c-np.array((0,2))), tuple(c+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c-np.array((2,0))), tuple(c+np.array((2,0))), (255,255,0), 1)

        c = positions[2], positions[3]
        cv2.line(detection, tuple(c-np.array((0,2))), tuple(c+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c-np.array((2,0))), tuple(c+np.array((2,0))), (255,255,0), 1)


        plt.imshow(detection, cmap='gray')
        plt.show()
        #plt.pause(0.01)

if __name__ == '__main__':
    #main()
    test_dataset()
    
