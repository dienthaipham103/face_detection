import cv2

'''
Draw boxes on img
xmin------------>xmax
ymin
|
|
|
V
ymax
'''

def draw_boxes(img, boxes):
    '''
    Draw boxes on img
    :img: bgr img array
    :boxes: xmin, ymin, xmax, ymax
    '''
    for (xmin, ymin, xmax, ymax) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return img