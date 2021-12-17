from retinaface import RetinaFace
# from pathlib import Path

resp = RetinaFace.detect_faces("test.jpg")
# print(resp)

boxes = []
for k in resp.keys():
    print(k)
    boxes.append(resp[k]['facial_area'])

print(boxes)

# print(Path.home())