from retinaface.retinaface import RetinaFace
import cv2
import argparse
from draw_boxes import *

'''
Original source: https://github.com/serengil/retinaface
'''

def retinaface_detector(img):
    resp = RetinaFace.detect_faces(img)

    boxes = []
    for k in resp.keys():
        boxes.append(resp[k]['facial_area'])

    return boxes

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
        help="path to input image")
    args = vars(ap.parse_args())

    # BGR to RGB channel ordering (which is what dlib expects)
    path = args["image"]
    image = cv2.imread(path)
    boxes = retinaface_detector(image)
    image = draw_boxes(image, boxes)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    cv2.imwrite('output1.png', image)
