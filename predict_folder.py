import os
import cv2
import time
# from dlib_hog import *
# from dlib_cnn import *
# from mtcnn import *
# from mtcnn_detect import mtcnn_detector
# from ssd_caffe_detect import *
from ssh_detect import *
from retinaface_detect import *

'''
Draw face boxes on images
'''
detect_types = {'ssh': ssh_detect, 'retinaface': retinaface_detector}
# detect_types = {'dlib_hog': hog_face_detector, 'dlib_cnn': cnn_face_detector, 'mtcnn': mtcnn_detector, 
#                 'ssd': ssd_caffe_detector, 'ssh': ssh_detect, 'retinaface': retinaface_detector}
# for detect_type in detect_types.keys():
#     type_folder = os.path.join('input', detect_type)
#     os.mkdir(type_folder)


start = time.time()
c = 0
for filename in os.listdir('input'):
    c += 1
    print(c)

    path = os.path.join('input', filename)

    for k in detect_types.keys():
        # BGR to RGB channel ordering (which is what dlib expects)
        image = cv2.imread(path)
        saved_path = os.path.join('output', filename[:-4] + '_' + k + '.png')
        boxes = detect_types[k](image)
        image = draw_boxes(image, boxes)

        # show the output image
        cv2.imwrite(saved_path, image)

end = time.time()
print('Duration: ', end - start)