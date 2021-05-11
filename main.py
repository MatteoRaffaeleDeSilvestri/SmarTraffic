import cv2
import numpy
import multiprocessing
import json
import time
import copy
import os
        
def run(source, dp, sts):

    # Prepare the object recognition system (YOLOv4)
    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    
    classes = list()
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()

    video = Video(net, classes, source, dp, sts)

    # Initialize process
    play = multiprocessing.Process(target=video.play)
    detect = multiprocessing.Process(target=video.detector)

    # Starting multiprocessing procedure
    detect.start()
    play.start()

    timeout = 5
    while True:
        if play.exitcode == 0 and not len(os.listdir('tmp')):
            os.kill(detect.pid, 9)
            timeout = 0
            break
        time.sleep(timeout)

class Video:

    def __init__(self, net, classes, source, detection_point, stats):

        self.net = net
        self.classes = classes
        self.camera = source
        self.detection_point = detection_point
        self.stats = stats

    def play(self):

        # Import camera settings
        with open('CAMERA_SETTINGS.json', 'r') as f:
            CAMERA_SETTINGS = json.load(f)

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
        object_detector = cv2.createBackgroundSubtractorMOG2(history=CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][0], varThreshold=CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][1])

        while True:

            ret, frame = capture.read()

            if not ret:
                capture.release()
                cv2.destroyAllWindows()
                break
            
            start_update = time.time()

            original = copy.deepcopy(frame)

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
            if self.detection_point:
                cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (0, 0, 255), 2)
            
            if passing_SX:
                if not os.path.isfile('tmp/{}.png'.format(str(vehicle_ID) + '_IN')):
                    vehicle_count_SX += 1
                    cv2.imwrite('tmp/{}.png'.format(str(vehicle_ID) + '_IN'), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][1],
                                                                                       CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][3]])
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            if passing_DX:
                if not os.path.isfile('tmp/{}.png'.format(str(vehicle_ID) + '_OUT')):
                    vehicle_count_DX += 1
                    cv2.imwrite('tmp/{}.png'.format(str(vehicle_ID) + '_OUT'), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][1],
                                                                                        CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][3]])
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
                
            if not passing_SX and not passing_DX:
                vehicle_ID += 1
            
            passing_SX = False
            passing_DX = False
            
            # Wait for ESC key to stop
            if cv2.waitKey(30) == 27:
                capture.release()
                cv2.destroyAllWindows()
                break

            # Update data on the screen (almost) every second
            end_update = time.time()
            update_interval += end_update - start_update
            if update_interval >= 1:
                FPS = int(1 / round(end_update - start_update, 3))
                update_interval = 0
            
            # display live statistics on the screen 
            if self.stats:
                cv2.rectangle(frame, (5, 5), (250, 110), (145, 145, 145), -1)
                cv2.putText(frame, 'FPS: {}'.format(FPS), (8, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)
                cv2.putText(frame, 'In ingresso: {}'.format(vehicle_count_SX), (8, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)
                cv2.putText(frame, 'In uscita: {}'.format(vehicle_count_DX), (8, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)
                cv2.putText(frame, 'Previsioni traffico: {}'.format('scorrevole'), (8, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)

            # Show the video (and layer)
            cv2.imshow(CAMERA_SETTINGS[source[6 : len(source) - 4]]["Title"], frame)
            # cv2.imshow('Left lane', mask_SX)
            # cv2.imshow('Right lane', mask_DX)

    def detector(self):

        detecting = True

        while detecting:
            
            if len(os.listdir('tmp')):

                base = cv2.imread('ticket.png')

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
                        
                            scores = detection[5 : ]
                            class_id = numpy.argmax(scores)
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
                    
                    if len(indexes) == 1:

                        ticket = base.copy()
                        
                        i = indexes.flatten()[0]
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        confidence = str(round(confidences[i] * 100, 1)) + '%'
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # cv2.rectangle(img, (x - 1, y), (x + w + 1, y - 20), (0, 255, 0), -1)
                        ticket[18 : 427, 120 : 370] = img[:]
                        cv2.putText(ticket, label + ' ' + confidence, (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
                        cv2.putText(ticket, '', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        print('Multiple or no object detection', len(indexes))
                        pass

                    # Save new (generated) photo
                    cv2.imwrite('detections/{}.png'.format(len(os.listdir('detections')) + 1), ticket)
                    
                    # Delete analized photos
                    os.remove('tmp/{}'.format(photo))
                    
            else:
                ''' Set a dynamic value for sleep '''
                time.sleep(5)

#     def traffic_conditions(self):
#         # Determina le condizioni del traffico
    
def ticket_maker(self, obj, conf):
    
    base = cv2.imread('ticket.png')
    auto = cv2.imread('detections/1.png')
    ticket = base.copy()
    ticket[27 : 427, 120 : 370] = auto[0 : 400 , 0 : 250]

    return ticket

#     def speed_detector(self):
#         # Determina le condizioni del traffico

run('video/camera_1.mp4', 1, 1)