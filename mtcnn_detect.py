import cv2
import argparse
from mtcnn.mtcnn import MTCNN
from draw_boxes import draw_boxes

detector = MTCNN()

def mtcnn_detector(img):
    location = detector.detect_faces(img)
    boxes = [face['box'] for face in location]

    # Get list of boxes (xmin, ymin, xmax, ymax)
    boxes = [(x, y, x+w, y+h) for (x, y, w, h) in boxes]

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
    boxes = mtcnn_detector(image)
    image = draw_boxes(image, boxes)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)