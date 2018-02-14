import numpy as np
import cv2

def open_image(path):
    """ returns the grayscale image"""

    # Open file
    with open(path, 'rb') as f:
        # Reading header
        file_format = f.readline().decode("utf-8")[:-1]
        width    = int(f.readline().decode("utf-8")[:-1])
        height   = int(f.readline().decode("utf-8")[:-1])
        max_gray = int(f.readline().decode("utf-8")[:-1])

        #print("file format:", file_format)
        #print("width:", width)
        #print("height:", height)
        #print("max gray:", max_gray)

        img = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                b = f.read(1)
                if b != b'\n':
                    p = int.from_bytes(b, 'big')
                    img[i][j] = p

        #plt.imshow(detection)
        #plt.show()

    return img
                
def open_eye_pos(path):
    """ returns groundtruth eyes positions
    format is: (left x, left y, right x, right y)
    """ 

    # Open file
    with open(path, 'r') as f:
        positions = list(f)[1]
        positions = positions.split('\t')
        positions[-1] = positions[-1][:-1]
        positions = [int(p) for p in positions]

    return tuple(positions)


if __name__ == '__main__':
    #open_image('BioID-FaceDatabase-V1.2/BioID_0001.pgm')
    open_eye_pos('BioID-FaceDatabase-V1.2/BioID_0001.eye')

