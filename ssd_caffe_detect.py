import cv2
import numpy as np
import argparse
from draw_boxes import *

# load our serialized face detector model from disk
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

def ssd_caffe_detector(img, threshold=0.5):
    (h, w) = img.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            boxes.append((startX, startY, endX, endY))
    
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
    boxes = ssd_caffe_detector(image)
    # boxes = cnn_face_detector(image)
    image = draw_boxes(image, boxes)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)