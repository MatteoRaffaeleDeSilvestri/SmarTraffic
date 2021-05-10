import cv2
import numpy as np
from PIL import Image
from time import time, sleep
import os
import multiprocessing
import json

# Global variable
DETECTION_POINT = True
CAMERA_SETTINGS = None

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
        capture = cv2.VideoCapture(source)
       
        # Cam variables
        passing_SX = False
        passing_DX = False
        vehicle_ID = 0
        vehicle_count_SX = 0
        vehicle_count_DX = 0
        update_interval = 0
        FPS = '-'

        # Object detection from camera
        object_detector = cv2.createBackgroundSubtractorMOG2(CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][1])

        while True:

            _, frame = capture.read()

            if frame is None:
                capture.release()
                cv2.destroyAllWindows()
                os.kill(show.pid, 9)
            
            start_update = time()

            original = frame

            # Define region of interest (ROI)
            roi_SX = frame[CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][1],
                           CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][3]]
            roi_DX = frame[CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][1],
                           CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][3]]

            # Generate mask for object
            mask_SX = object_detector.apply(roi_SX)
            mask_DX = object_detector.apply(roi_DX)

            # Remove shadow and expand detection in the image
            _, mask_SX = cv2.threshold(mask_SX, 254, 255, cv2.THRESH_BINARY)
            _, mask_DX = cv2.threshold(mask_DX, 254, 255, cv2.THRESH_BINARY)

            # Find contours for moving object
            contours_SX, _ = cv2.findContours(mask_SX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_DX, _ = cv2.findContours(mask_DX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            for contours in contours_SX:

                # Calculate areas and remove small elements
                if cv2.contourArea(contours) >= CAMERA_SETTINGS[source[6 : len(source) - 4]]["Surface"]:
                    
                    x, y, w, h = cv2.boundingRect(contours)

                    # Animate detection point
                    if CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1] in range(CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] + y, CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] + y + h):
                        passing_SX = True

            for contours in contours_DX:

                # Calculate areas and remove small elements
                if cv2.contourArea(contours) >= CAMERA_SETTINGS[source[6 : len(source) - 4]]["Surface"]:
                    
                    x, y, w, h = cv2.boundingRect(contours)

                    # Animate detection point
                    if CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1] in range(CAMERA_SETTINGS[source[6 :len(source) - 4]]["ROI_SX"][0] + y, CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] + y + h):
                        passing_DX = True

            # Draw detection point
            if DETECTION_POINT:
                cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (0, 0, 255), 2)
            
            if passing_SX:
                if not os.path.isfile('tmp/{}.png'.format(str(vehicle_ID) + '_IN')):
                    vehicle_count_SX += 1
                    cv2.imwrite('tmp/{}.png'.format(str(vehicle_ID) + '_IN'), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][1],
                                                                                       CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][3]])
                if DETECTION_POINT:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            if passing_DX:
                if not os.path.isfile('tmp/{}.png'.format(str(vehicle_ID) + '_OUT')):
                    vehicle_count_DX += 1
                    cv2.imwrite('tmp/{}.png'.format(str(vehicle_ID) + '_OUT'), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][1],
                                                                                        CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][3]])
                if DETECTION_POINT:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
                
            if not passing_SX and not passing_DX:
                vehicle_ID += 1
            
            passing_SX = False
            passing_DX = False
            
            # Wait for ESC key to stop
            key = cv2.waitKey(30)
            if key == 27:
                capture.release()
                cv2.destroyAllWindows()
                os.kill(show.pid, 9)

            # Update data on the screen (almost) every second
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
            cv2.imshow(CAMERA_SETTINGS[source[6 : len(source) - 4]]["Title"], frame)
            cv2.imshow('Left lane', mask_SX)
            cv2.imshow('Right lane', mask_DX)

    def detector(self):

        while True:
            
            if len(os.listdir('tmp')):

                for photo in os.listdir('tmp'):
                    
                    img = cv2.imread('tmp/{}'.format(photo))
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
                    os.remove('tmp/{}'.format(photo))
                    
            else:
                ''' Set a dynamic value for sleep '''
                sleep(5)

if __name__ == '__main__':

    # Take video frome source (GUI)
    source = 'video/camera_1.mp4'

    # Import camera settings
    with open('CAMERA_SETTINGS.json', 'r') as f:
        CAMERA_SETTINGS = json.load(f)

    vid = Video(source)
    
    # Initialize process
    show = multiprocessing.Process(target=vid.play)
    detect = multiprocessing.Process(target=vid.detector)

    # Starting multiprocessing procedure
    detect.start()
    show.start()

    while True:
        ''' Ottimizza questo ciclo ''' 
        if not show.is_alive() and not len(os.listdir('tmp')):
            os.kill(detect.pid, 9)
            break
        sleep(5)
