from dlib_hog import *
from retinaface_detect import *
import cv2
from draw_boxes import draw_boxes

# Create an object to read from camera
video = cv2.VideoCapture("dien.mp4")

# We need to check if camera is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.So, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
result = cv2.VideoWriter('dien_output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 29, size)
frame_num=0
while (True):
    ret, frame = video.read()
    frame_num += 1
    print(frame_num)
    if ret == True:

        boxes = retinaface_detector(frame)
        frame = draw_boxes(frame, boxes)
        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break


video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")