import cv2
import numpy as np
from PIL import Image
from time import time, sleep
import os
import multiprocessing

class Video:

    def __init__(self, net, classes, source):

        self.net = net
        
        self.classes = classes
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()

        self.camera = source
         
    def play(self):

        # Take video frome source 
        source = self.camera
        cap = cv2.VideoCapture(source)
       
        # Cam variables
        passing_on = False
        car_id = 0
        count = 0
        update_interval = 0
        FPS = '-'

        # Object detection from camera
        object_detector = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=180)

        while True:

            _, frame = cap.read()

            if frame is None:
                print("Video ended")
                break
            
            start_update = time()

            frame = cv2.resize(frame, (1280, 720))
            original = cv2.resize(frame, (1280, 720))
            
            # Define region of interest (ROI)
            roi = frame[150 : 710, 390 : 670]

            # Generate mask for object
            mask = object_detector.apply(roi)

            # Remove shadow and expand detection in the image
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5)))
            
            # Find contours for moving object
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours
            for cnt in contours:

                # Calculate areas and remove small elements
                area = cv2.contourArea(cnt)

                if 1000 < area < 100000:
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) 
                    # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Animate detection point
                    if 600 in range(150 + y, 150 + y + h):
                        passing_on = True
            
            if passing_on:
                if not os.path.isfile('detections/{}.png'.format(car_id)):
                    count += 1
                    cv2.imwrite('detections/{}.png'.format(car_id), original[150 : 710, 400 : 700])
                cv2.line(frame, (455, 600), (640, 600), (255, 255, 255), 2)
                passing_on = False
            else:
                cv2.line(frame, (455, 600), (640, 600), (0, 0, 255), 2)
                car_id += 1

            # Wait foe ESC key to stop
            key = cv2.waitKey(30)
            if key == 27:
                break

            # Update data on the screen every second
            end = time()
            update_interval += end - start_update
            if update_interval >= 1:
                FPS = int(1 / round(end - start_update, 3))
                update_interval = 0
            
            cv2.putText(frame, 'FPS: {}'.format(FPS), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            cv2.putText(frame, 'Vehicles: {}'.format(count), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

            # Show the video (and layer)
            # cv2.imshow(camera[6 : len(camera) - 4], frame)
            cv2.imshow(source[6 : len(source) - 4], frame)
            # cv2.imshow('ROI {}'.format(camera[6 : len(camera) - 4]), mask)
            
        cap.release()
        cv2.destroyAllWindows()
        capture.close()

    def detector(self):

        while True:
            
            if len(os.listdir('detections')):

                for photo in os.listdir('detections'):

                    # start = time()

                    img = cv2.imread('detections/{}'.format(str(photo)))
                    height, width, _ = img.shape
                    # print(height, width)

                    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True, False)

                    # for b in blob:
                    #     for n, img_blob in enumerate(b):
                    #         cv2.imshow(str(n), img_blob)

                    self.net.setInput(blob)

                    output_layers_names = self.net.getUnconnectedOutLayersNames()
                    layerOutputs = self.net.forward(output_layers_names)

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
                            label = str(self.classes[class_ids[i]])
                            confidence = str(round(confidences[i] * 100, 1)) + '%'
                            color = colors[i]
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            img = cv2.rectangle(img, (x - 1, y), (x + w + 1, y - 20), (0, 255, 0), -1)
                            img = cv2.putText(img, label + ' ' + confidence, (x, y - 5), font, 1, (10, 10, 10), 1)
                            cv2.putText(img, '', (x, y + 20), font, 2, (255, 255, 255), 1)
                            # print(x, y, w, h)
                            cv2.imwrite('analysed/{}'.format(photo), img[y - 30 : y + h + 30, x - 30 : x + w + 30])
                    
                    os.remove('detections/{}'.format(photo))
                            
                    # cv2.imshow('Image', img)
                    # end = time()
                    # print(end - start)
                    
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            
            else:
                # print('Empty folder')
                sleep(5)

if __name__ == '__main__':

    # Prepare the object recognition system (YOLOv4)
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    
    classes = list()
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()

    
    source = 'video/drone.mp4'
    
    vid = Video(net, classes, source)
    
    # Initialize process
    capture = multiprocessing.Process(target=vid.play)
    detect = multiprocessing.Process(target=vid.detector)

    # Starting multiprocessing procedure
    detect.start()
    capture.start()

    while True:
        if not capture.is_alive():
            detect.kill()
            break
