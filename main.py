import cv2
import numpy as np
from PIL import Image
from time import time, sleep
import os
import multiprocessing

class Video:

    def __init__(self, source):

        # Prepare the object recognition system (YOLOv4)
        self.net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    
        self.classes = list()
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()

        self.camera = source
         
    def play(self):

        # Take video frome source 
        source = self.camera
        cap = cv2.VideoCapture(source)
       
        # Cam variables
        passing_SX = False
        passing_DX = False
        vehicle_ID_SX = 0
        vehicle_ID_DX = 0
        vehicle_count_SX = 0
        vehicle_count_DX = 0
        update_interval = 0
        FPS = '-'

        # Object detection from camera
        object_detector = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=180)

        while True:

            _, frame = cap.read()

            if frame is None:
                cap.release()
                cv2.destroyAllWindows()
                os.kill(show.pid, 9)
            
            start_update = time()

            frame = cv2.resize(frame, (1280, 720))
            original = cv2.resize(frame, (1280, 720))
            
            # Define region of interest (ROI)
            roi_SX = frame[300 : 700, 400 : 650]
            roi_DX = frame[300 : 700, 650 : 900]

            # Generate mask for object
            mask_SX = object_detector.apply(roi_SX)
            mask_DX = object_detector.apply(roi_DX)

            # Remove shadow and expand detection in the image
            _, mask_SX = cv2.threshold(mask_SX, 254, 255, cv2.THRESH_BINARY)
            mask_SX = cv2.dilate(mask_SX, np.ones((5, 5)))
            _, mask_DX = cv2.threshold(mask_DX, 254, 255, cv2.THRESH_BINARY)
            mask_DX = cv2.dilate(mask_DX, np.ones((5, 5)))
            
            # Find contours for moving object
            contours_SX, _ = cv2.findContours(mask_SX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_DX, _ = cv2.findContours(mask_DX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            for contours in contours_SX:

                # Calculate areas and remove small elements
                if cv2.contourArea(contours) >= 1000:
                    
                    x, y, w, h = cv2.boundingRect(contours)
                    # cv2.drawContours(roi, [contours], -1, (0, 255, 0), 2) 
                    # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Animate detection point
                    if 500 in range(300 + y, 300 + y + h):
                        passing_SX = True
        
            for contours in contours_DX:

                # Calculate areas and remove small elements
                if cv2.contourArea(contours) >= 1000:
                    
                    x, y, w, h = cv2.boundingRect(contours)
                    # cv2.drawContours(roi, [contours], -1, (0, 255, 0), 2) 
                    # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Animate detection point
                    if 500 in range(300 + y, 300 + y + h):
                        passing_DX = True

            cv2.line(frame, (485, 500), (810, 500), (0, 0, 255), 2)
            if passing_SX:
                if not os.path.isfile('detections/{}.png'.format(str(vehicle_ID_SX) + '_IN')):
                    vehicle_count_SX += 1
                    cv2.imwrite('detections/{}.png'.format(str(vehicle_ID_SX) + '_IN'), original[300 : 700, 400 : 650])
                cv2.line(frame, (485, 500), (647, 500), (255, 255, 255), 2)
            
            if passing_DX:
                if not os.path.isfile('detections/{}.png'.format(str(vehicle_ID_DX) + '_OUT')):
                    vehicle_count_DX += 1
                    cv2.imwrite('detections/{}.png'.format(str(vehicle_ID_DX) + '_OUT'), original[300 : 700, 650 : 900])
                cv2.line(frame, (647, 500), (810, 500), (255, 255, 255), 2)
                
            if not passing_SX and not passing_DX:
                vehicle_ID_SX += 1
                vehicle_ID_DX += 1
            
            passing_SX = False
            passing_DX = False
            
            # Wait foe ESC key to stop
            key = cv2.waitKey(30)
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                os.kill(show.pid, 9)

            # Update data on the screen every second
            end_update = time()
            update_interval += end_update - start_update
            if update_interval >= 1:
                FPS = int(1 / round(end_update - start_update, 3))
                update_interval = 0
            cv2.rectangle(frame, (5, 5), (250, 110), (50, 50, 50), -1)
            cv2.putText(frame, 'FPS: {}'.format(FPS), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'Vehicle SX: {}'.format(vehicle_count_SX), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'Vehicle DX: {}'.format(vehicle_count_DX), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # Show the video (and layer)
            # cv2.imshow('Left lane', mask_SX)
            # cv2.imshow('Right lane', mask_DX)
            cv2.imshow(source[6 : len(source) - 4], frame)

    def detector(self):

        while True:
            
            if len(os.listdir('detections')):

                for photo in os.listdir('detections'):
                    
                    img = cv2.imread('detections/{}'.format(photo))
                    height, width, _ = img.shape

                    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), True, False)
 
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

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    
                    if len(indexes) > 0:
                        for i in indexes.flatten():
                            x, y, w, h = boxes[i]
                            label = str(self.classes[class_ids[i]])
                            confidence = str(round(confidences[i] * 100, 1)) + '%'
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.rectangle(img, (x - 1, y), (x + w + 1, y - 20), (0, 255, 0), -1)
                            cv2.putText(img, label + ' ' + confidence, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 1)
                            cv2.putText(img, '', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)

                    # Save new (generated) photo
                    cv2.imwrite('analysed/{}.png'.format(len(os.listdir('analysed')) + 1), img)
                    
                    # Delete analized photos
                    os.remove('detections/{}'.format(photo))
                    
            else:
                ''' Set a dynamic value for sleep '''
                sleep(5)

if __name__ == '__main__':

    # Take video frome source (GUI)
    source = 'video/drone.mp4'
    
    vid = Video(source)
    
    # Initialize process
    show = multiprocessing.Process(target=vid.play)
    detect = multiprocessing.Process(target=vid.detector)

    # Starting multiprocessing procedure
    detect.start()
    show.start()

    while True:
        ''' Ottimizza questo ciclo ''' 
        if not show.is_alive() and not len(os.listdir('detections')):
            os.kill(detect.pid, 9)
            break
        sleep(5)
