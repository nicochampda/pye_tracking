import gc
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from open_image import open_image
from open_image import open_eye_pos

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_eye(frame, is_gray = False):

    global face_cascade
    global eye_cascade

    gray = None
    if is_gray:
        gray = np.copy(frame)
    else:
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray)

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

        # Extraction des deux yeux les plus hauts
        if len(eyes) > 2:
            eyes2 = [list(i) for i in eyes]
            eyes2.sort(key=lambda tup: tup[1])
            eyes_haut = [eyes2[0], eyes2[1]]
        else:
            eyes_haut = eyes

        centres = []
        for (ex,ey,ew,eh) in eyes_haut:

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #Region d'interet autour des yeux
            roi_eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            roi_eye_color = roi_color[ey:ey+eh, ex:ex+ew]
            #print(roi_eye_gray.shape)
            #if np.min(roi_eye_gray.shape) > 5:
            c = detect_eye_center(roi_eye_gray)
            #c = ( np.int(eh/2),np.int(ew/2))
            #print(ey, ey+eh, ex, ex+ew, roi_eye_color.shape, roi_color.shape,c)
            cv2.line(roi_eye_color, tuple(c-np.array((0,2))), tuple(c+np.array((0,2))), (0,255,255), 1)
            cv2.line(roi_eye_color, tuple(c-np.array((2,0))), tuple(c+np.array((2,0))), (0,255,255), 1)

            centres.append(np.array(c) + [ex + x, ey + y])


        # Tri: gauche puis droite
        centres.sort(key=lambda tup: tup[0])
                 
    return frame, centres 


def detect_eye_center(img):
    #print(img.shape)
    grad_x = np.gradient(img, axis=0)
    grad_y = np.gradient(img, axis=1)

    # Heatmap des gradients
    plt.subplot(141)
    plt.imshow(grad_x, cmap = 'hot')
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(img, cmap='gray')
    plt.subplot(143)
    plt.imshow(grad_y, cmap='hot')
    plt.colorbar()
    plt.subplot(144)
    plt.imshow((grad_x + grad_y) / 2, cmap='hot')
    plt.colorbar()
    plt.show()

    # Histogramme des gradients
    #plt.subplot(121)
    #plt.hist(grad_x, bins=256)
    #plt.subplot(122)
    #plt.hist(grad_y, bins=256)
    #plt.show()

    x,y = img.shape
    c = 0,0
    max_val = 0
    seuil = 230
    less_x, less_y = np.where((grad_x > seuil) | (grad_y > seuil))
    #print(less_x.shape, less_y.shape)
    for i in range(x):
        #print(i)
        for j in range(y):
            somme = 0
            for nb, k in enumerate(less_x):
                l = less_y[nb]
                if (i,j) != (k,l):
                    dx = k - i
                    dy = l - j
                    norm = np.sqrt(dx**2 + dy**2)
                    somme += (grad_x[k][l]*(dx/norm) + grad_y[k][l]*(dy/norm))**2
                
            if somme > max_val:
                max_val = somme
                c = i,j
    return c
    

def test_dataset():
    for i in range(30):

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

        # Estimation du centre des yeux
        detection, centres = detect_eye(img, is_gray = True)

        #print("estimes:", centres)
        #print("verite :", positions)

        # Centre des yeux gauche et droit
        c_gt_r = positions[0], positions[1]
        c_gt_l = positions[2], positions[3]

        # Dessin des centres verite terrain
        cv2.line(detection, tuple(c_gt_l-np.array((0,2))), tuple(c_gt_l+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_l-np.array((2,0))), tuple(c_gt_l+np.array((2,0))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_r-np.array((0,2))), tuple(c_gt_r+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_r-np.array((2,0))), tuple(c_gt_r+np.array((2,0))), (255,255,0), 1)

        distances = []

        if len(centres) == 2:
            # Calcul de la distance
            dist_l = np.trunc(distance_center(c_gt_l, centres[0]) * 100) / 100
            dist_r = np.trunc(distance_center(c_gt_r, centres[1]) * 100) / 100

            distances.append(dist_l)
            distances.append(dist_r)

            # Affichage des distances
            font = cv2.FONT_HERSHEY_SIMPLEX
            orig_l = c_gt_l[0], c_gt_l[1] + 20
            cv2.putText(detection, str(dist_l), orig_l, font, 0.3, (255,255,255))

            orig_r = c_gt_r[0], c_gt_r[1] + 20 # MAUVAIS SENS ?
            cv2.putText(detection, str(dist_r), orig_r, font, 0.3, (255,255,255))

            plt.imshow(detection, cmap='gray')
            #plt.pause(0.2)
            plt.show()
            gc.collect()

    moyenne = np.mean(distances)
    print(moyenne)



def distance_center(pos_gt, pos_est):
    """ doit etre ndarray"""
    return np.sqrt(np.sum((pos_gt - pos_est) ** 2))


def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv.CV_CAP_PROP_FPS, 1) 
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        detection, centres = detect_eye(frame)
    
        # Display the resulting frame
        cv2.imshow('frame', detection)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    #test_dataset()
