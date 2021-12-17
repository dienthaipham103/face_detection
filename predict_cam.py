import cv2
from retinaface_detect import *
from draw_boxes import draw_boxes


video = cv2.VideoCapture(0)

if (video.isOpened() == False):
    print("Web Camera not detected")
while (True):
    ret, frame = video.read()
    if ret == True:
        boxes = retinaface_detector(frame)
        frame = draw_boxes(frame, boxes)
        cv2.imshow("Output",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()