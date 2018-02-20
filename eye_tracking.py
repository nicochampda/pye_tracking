import gc
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from open_image import open_image
from open_image import open_eye_pos

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_eye.xml')
 
args = None

def detect_eye(frame, is_gray = False):
    """ Detecte le visage, la zone des yeux et la pupille d'une image
    :param frame: image a traiter
    :param is_gray: indique si frame est une image en niveaux de gris
    :type frame: ndarray
    :type is_gray: bool
    :return: un tuple frame, centres. frame est l'image ou la detection a
             ete faite et ou les rectangles de detection ont ete dessines.
             centres est un tableau contenant les coordonnees des centres
             des yeux. Il est donc de taille 2 et le premier element est
             les coordonnees de l'oeil a gauche de l'image.
    :rtype: tuple (ndarray, array(x, y))
    """


    global args

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

    centres = []
    # Pour tous les visages detectees
    for (x,y,w,h) in faces:
        # Dessin d'un rectangle autour du visage
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

        # Regions d'interet du visage
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Extraction des yeux
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Extraction des deux yeux les plus hauts
        if len(eyes) > 2: # seulement si on detecte plus de deux yeux
            eyes2 = [list(i) for i in eyes]
            eyes2.sort(key=lambda tup: tup[1])
            eyes_haut = [eyes2[0], eyes2[1]]
        else:
            eyes_haut = eyes

        centres = []
        # Pour chaque oeil detecte
        for (ex,ey,ew,eh) in eyes_haut:

            # Dessin d'un rectangle autour de l'oeil
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            #Region d'interet autour des yeux
            roi_eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            roi_eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Detection du centre de la pupille
            c = detect_eye_center(roi_eye_gray, w, ew)

            # Valeur reference: milieu du rectangle
            #c = ( np.int(eh/2),np.int(ew/2))

            # Dessin d'une croix au centre
            cv2.line(roi_eye_color, tuple(c-np.array((0,50))),
                    tuple(c+np.array((0,50))), (0,255,255), 1)
            cv2.line(roi_eye_color, tuple(c-np.array((50,0))),
                    tuple(c+np.array((50,0))), (0,255,255), 1)

            # Enregistrement des centres calcules
            # relativement a l'image totale 
            centres.append(np.array(c) + [ex + x, ey + y])

        # Tri: gauche puis droite
        centres.sort(key=lambda tup: tup[0])
                 
    return frame, centres 


def detect_eye_center(img, face_size, eye_size):
    """ Calcule le centre optimal de l'oeil
    :param img: image de la zone autour de l'oeil
    :param face_size: taille du visage
    :param eye_size: taille de la zone des yeux
    :type img: ndarray
    :type face_size: int
    :type eye_size: int
    :return: coordonnees du centre de l'oeil
             (relativement a la region d'interet de l'oeil)
    :rtype: tuple (i, j)
    """

    global args

    # Calcul des gradients dans les deux directions
    grad_x = np.gradient(img, axis=0)
    grad_y = np.gradient(img, axis=1)
    
    # Histogramme des gradients
    #plt.subplot(121)
    #plt.hist(grad_x, bins=256)
    #plt.subplot(122)
    #plt.hist(grad_y, bins=256)
    #plt.show()

    x,y = img.shape
    
    # Padding
    grad_x[0][:]  = np.zeros(y)    
    grad_x[-1][:] = np.zeros(y)    
    grad_y[0][:]  = np.zeros(x)   
    grad_y[-1][:] = np.zeros(x)
    
    c = 0,0
    max_val = 0
    # Seuillage des gradients
    less_x, less_y = np.where((grad_x > args.seuil) | (grad_y > args.seuil))
    less = [(less_x[i], less_y[i]) for i in range(len(less_x))]

    # Heatmap des gradients
    plt.figure()
    plt.subplot(141)
    plt.imshow(grad_x, cmap = 'jet')
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(grad_y, cmap='jet')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow((grad_x + grad_y) / 2, cmap='jet')
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(img, cmap='gray')
    plt.show()

    # Delimite la zone où les centres potentiels sont recherches
    border = int((eye_size - (face_size / 9)) / 2) 
    
    # Parcours de tous les points susceptibles
    # d'etre le centre optimal
    for i in range(border, x-border):
        for j in range(border, y-border):
            somme = 0
            try:
                less.remove((i,j))
                flag = True
            except ValueError:
                flag = False

            pix_val = 255 - img[i][j]

            for k,l in less:
                dx = k - i
                dy = l - j
                norm = np.sqrt(dx**2 + dy**2)
                somme += pix_val * ((grad_x[k][l]*dx + grad_y[k][l]*dy)/norm)**2
            
            if flag:
                less.append((i,j))
            
            if somme > max_val:
                max_val = somme
                c = i,j
    return c

def hist_photo():
    """ Test de l'algorithme sur une image unique issue d'une webcam """
    # Image test
    img = cv2.imread('photo2.jpg', 0)

    # Detection
    detection, centres = detect_eye(img, is_gray = True)

    # Affichage du resultat
    plt.imshow(detection)
    plt.show()
    

def test_dataset():
    """ Permet de tester l'algorithme sur le dataset BioID"""

    global args

    distances = []
    errors = []
    accepte = 0

    for i in range(args.nb_img):

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

        # Centre des yeux gauche et droit
        c_gt_r = positions[0], positions[1]
        c_gt_l = positions[2], positions[3]

        # Estimation du centre des yeux
        detection, centres = detect_eye(img, is_gray = True)

        # Dessin des centres verite terrain
        cv2.line(detection, tuple(c_gt_l-np.array((0,2))),
                tuple(c_gt_l+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_l-np.array((2,0))),
                tuple(c_gt_l+np.array((2,0))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_r-np.array((0,2))),
                tuple(c_gt_r+np.array((0,2))), (255,255,0), 1)
        cv2.line(detection, tuple(c_gt_r-np.array((2,0))),
                tuple(c_gt_r+np.array((2,0))), (255,255,0), 1)

        # Si deux yeux ont ete detectes
        if len(centres) == 2:

            # Compte d'images ou la detection a ete faite
            accepte += 1

            # Calcul de la distance
            dist_l = np.trunc(distance_center(c_gt_l, centres[0]) * 100) / 100
            dist_r = np.trunc(distance_center(c_gt_r, centres[1]) * 100) / 100
            distances.append(dist_l)
            distances.append(dist_r)

            # Erreur moyenne
            d = distance_center(c_gt_l, np.array(c_gt_r))
            error = (dist_l + dist_r) / (2 * d)
            errors.append(error)

            # Dessin des distances
            font = cv2.FONT_HERSHEY_SIMPLEX
            orig_l = c_gt_l[0], c_gt_l[1] + 20
            cv2.putText(detection, str(dist_l), orig_l, font, 0.3, (255,255,255))

            orig_r = c_gt_r[0], c_gt_r[1] + 20 
            cv2.putText(detection, str(dist_r), orig_r, font, 0.3, (255,255,255))

            # Affichage
            if args.images:
                plt.imshow(detection, cmap='gray')
                #plt.pause(0.2)
                plt.show()
            gc.collect()

    # Statistiques sur les performances de l'agorithme
    moyenne = np.mean(distances)
    print("\n------ STATISTIQUES ------")
    print("moyenne:", moyenne)
    print("  total:", args.nb_img)
    print("accepte:", accepte)
    print(" rejete:", args.nb_img - accepte)
    print("--------------------------")

    plt.hist(errors, cumulative=True, histtype='step', bins=500)
    plt.show()

def distance_center(pos_gt, pos_est):
    """ Calcule la distance euclidienne entre deux coordonnees 
    :param pos_gt: coordonnees verite terrain
    :param pos_est: coordonnees calculees
    :type post_gt: tuple (i, j)
    :type post_est: tuple (i, j)
    :return: distance en pixel
    :rtype: float
    """
    return np.sqrt(np.sum((pos_gt - pos_est) ** 2))


def main():
    """ Test de l'algorithme sur un flux video venant de la webcam"""

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 2) # Ajustement du nombre de FPS (ne fonctionne pas)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Detection
        detection, centres = detect_eye(frame)
    
        # Display the resulting frame
        cv2.imshow('frame', detection)
    
        # Appuyer sur q pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Extinction de la camera et fermeture de la fenetre
    cap.release()
    cv2.destroyAllWindows()


def read_args():
    """ Options entrees en ligne de commande"""

    global args

    parser = argparse.ArgumentParser(description='Eye tracking.')

    parser.add_argument('-m', '--mode', action = 'store', default = 'webcam', choices = ['webcam', 'dataset', 'photo'],
            help = '\'webcam\' pour la detection sur un flux video de la webcam, \'photo\' pour une photo unique ou \'test\' sur le jeu de donnees BioID. \'webcam\' par defaut.')

    parser.add_argument('-n', '--nb_img', action = 'store', default = 50, type = int,
            help='Nombre d\'images a tester, 50 par defaut et 1520 max. Seulement avec le mode dataset.')

    parser.add_argument('-i', '--images', action = 'store_true', default = False,
            help = 'Affiche les detections sur les images tests. Desactive par defaut.')

    parser.add_argument('-s', '--seuil', action = 'store', default = 200, type = int,
            help='Valeur de seuil pour les gradients; doit etre compris entre 0 et 255. 200 par defaut.')


    args = parser.parse_args()

    if args.seuil > 255 or args.seuil < 0:
        print("Erreur: seuil invalide. Doit etre compris entre 0 et 255 inclus.")
        sys.exit(2)

    if args.nb_img > 1520 or args.nb_img < 1:
        print("Erreur: nombre d'images invalide. Doit etre compris entre 1 et 1520.")
        sys.exit(2)


if __name__ == '__main__':
    read_args()
    if args.mode == 'webcam':
        main()
    elif args.mode == 'dataset':
        test_dataset()
    elif args.mode == 'photo':
        hist_photo()
