import cv2
import numpy as np
from PIL import Image
from time import time, sleep
import os

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# print(classes)
while True:
    
    if len(os.listdir('detections')) == 0:
        print('No file to work on')
        sleep(5)
    else:        
    
        for photo in os.listdir('detections'):

            # start = time()

            img = cv2.imread('detections/{}'.format(str(photo)))
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
                    confidence = str(round(confidences[i] * 100, 1)) + '%'
                    color = colors[i]
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    img = cv2.rectangle(img, (x - 1, y), (x + w + 1, y - 20), (0, 255, 0), -1)
                    img = cv2.putText(img, label + ' ' + confidence, (x, y - 5), font, 1, (10, 10, 10), 1)
                    cv2.putText(img, '', (x, y + 20), font, 2, (255, 255, 255), 1)
                    # print(x, y, w, h)
                    cv2.imwrite('analysed/{}.png'.format(photo), img[y - 30 : y + h + 30, x - 30 : x + w + 30])
            
            os.remove('detections/{}'.format(photo))
                    
            # cv2.imshow('Image', img)
            # end = time()
            # print(end - start)
            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
