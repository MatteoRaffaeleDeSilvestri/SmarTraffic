import cv2
import numpy as np
from PIL import Image
from time import time

start = time()

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# print(classes)

img = cv2.imread('detections/detection0.png')
height, width, _ = img.shape
# print(height, width)

blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True, False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:

    for detection in output:
    
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
    
        if confidence > 0.5:
    
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
    
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
    
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

# print(len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# print(indexes.flatten())

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 1))
        color = colors[i]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 230, 0), 2)
        img = cv2.putText(img, label + ' ' + confidence, (x, y + 20), font, 1, (255, 255, 255), 1)
        cv2.putText(img, '', (x, y + 20), font, 2, (255, 255, 255), 1)
        cv2.imwrite('frame.jpg', img)

cv2.imshow('Image', img)
end = time()
print(end - start)
cv2.waitKey(0)
cv2.destroyAllWindows()
