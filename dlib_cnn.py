from utils import convert_and_trim_bb
import argparse
import imutils
import dlib
import cv2
from draw_boxes import draw_boxes

'''
dlib’s MMOD CNN face detector
'''

# Load dlib's CNN face detector
model="mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(model)

def cnn_face_detector(img, upsample=1, width=None):
    
    # resize image, and convert it from, save ratio to re-compute the boxes
    # BGR to RGB channel ordering (which is what dlib expects)
    if width is None:
        width = img.shape[1]
    resized_img = imutils.resize(img, width=width)
    ratio = img.shape[1]/width
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # perform face detection using dlib's face detector
    results = detector(rgb, upsample)

    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [convert_and_trim_bb(resized_img, r.rect) for r in results]

    # Get list of boxes (xmin, ymin, xmax, ymax)
    boxes = [(int(x*ratio), int(y*ratio), int((x + w)*ratio), int((y + h)*ratio)) for (x, y, w, h) in boxes]

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
    # boxes = cnn_face_detector(image, width=600)
    boxes = cnn_face_detector(image)
    image = draw_boxes(image, boxes)
    # print(boxes)

    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)