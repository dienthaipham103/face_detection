from utils import convert_and_trim_bb
import argparse
import imutils
import dlib
import cv2
from draw_boxes import draw_boxes

'''
dlib's HOG + Linear SVM face detector
'''

# Load dlib's HOG + Linear SVM face detector
detector = dlib.get_frontal_face_detector()

def hog_face_detector(img, upsample=1, width=None):
    # Resize image, and convert it from, save ratio to re-compute the boxes
    # BGR to RGB channel ordering (which is what dlib expects)
    if width is None:
        width = img.shape[1]
    resized_img = imutils.resize(img, width=width)
    ratio = img.shape[1]/width

    # img: bgr (read by cv2) -> rgb
    rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Perform face detection using dlib's face detector
    rects = detector(rgb, upsample)

    # Convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [convert_and_trim_bb(resized_img, r) for r in rects]

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
    boxes = hog_face_detector(image)
    # boxes = hog_face_detector(image, width=600)
    image = draw_boxes(image, boxes)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
