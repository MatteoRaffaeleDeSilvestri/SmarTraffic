import cv2
import numpy
import multiprocessing
import json
import time
import copy
import os
import calendar
import csv

# Import camera settings
with open('CAMERA_SETTINGS.json', 'r') as f:
    CAMERA_SETTINGS = json.load(f)

def start(source, dp, sts):

    # Prepare the object recognition system (YOLOv4)
    net = cv2.dnn.readNet('yolo/yolov4.weights', 'yolo/yolov4.cfg')
    
    classes = list()
    with open('yolo/coco.names', 'r') as f:
        classes = f.read().splitlines()

    # Delete existing ticket (if present)
    for ticket in os.listdir('detections')[:]:
        os.remove('detections/{}'.format(ticket))
    
    video = Video(net, classes, source, dp, sts)

    # Initialize process
    play = multiprocessing.Process(target=video.play)
    detect = multiprocessing.Process(target=video.detector)

    # Starting multiprocessing procedure
    play.start()
    detect.start()

    while True:
        if play.exitcode == 0 and not len(os.listdir('.tmp')):
            os.kill(detect.pid, 9)
            break
        time.sleep(3)

class Video:

    def __init__(self, ntt, clss, source, dp, sts):

        self.net = ntt
        self.classes = clss
        self.camera = source
        self.detection_point = dp
        self.stats = sts

        # Type of vehicle that can be seen on the road
        self.vehicles = {'Bicycle', 'Car', 'Motorbike', 'Bus', 'Truck', 'Boat'}

        # Common thing to be aware of during drive (other person and animals in general)
        self.other_object = {'Person', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Bear'}

    def play(self):

        # Take video frome source 
        source = self.camera
        capture = cv2.VideoCapture(source)
       
        # Camera variables
        passing_SX = False
        passing_DX = False
        vehicle_SX_ID = 0
        vehicle_DX_ID = 0
        vehicle_count_SX = 0
        vehicle_count_DX = 0
        update_interval = 0
        FPS = '-'
        
        # Set date and time
        date = CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"][ : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"].index('-') - 1].split('/')
        day = date[0]
        month = date[1]
        year = date[2]
        date = '{}/{}/{}'.format(day, month, year)
        
        clock = CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"][CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"].index('-') + 2 : ].split(':')
        hh = int(clock[0])
        mm = int(clock[1])
        ss = int(clock[2])
        clock = '{}:{}:{}'.format(hh, mm, ss)

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
            
            # Take a picture of the left lane (ROI)
            if passing_SX:
                if '{}.png'.format(str(vehicle_SX_ID) + '_C')[ : '{}'.format(str(vehicle_SX_ID) + '_C').index('C') + 1] not in [photo[ : '{}'.format(str(vehicle_SX_ID) + '_C').index('C') + 1] for photo in os.listdir('.tmp')]:
                    vehicle_count_SX += 1
                    cv2.imwrite('.tmp/{}.png'.format(str(vehicle_SX_ID) + '_C' + '{}{}{}{}{}{}'.format(day, month, year, hh, mm, ss)), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][1],
                                                                                                                                               CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][3]])
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            # Take a picture of the right lane (ROI)
            if passing_DX:
                if '{}.png'.format(str(vehicle_DX_ID) + '_L')[ : '{}'.format(str(vehicle_DX_ID) + '_L').index('L') + 1] not in [photo[ : '{}'.format(str(vehicle_SX_ID) + '_L').index('L') + 1] for photo in os.listdir('.tmp')]:
                    vehicle_count_DX += 1
                    cv2.imwrite('.tmp/{}.png'.format(str(vehicle_DX_ID) + '_L' + '{}{}{}{}{}{}'.format(day, month, year, hh, mm, ss)), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][1],
                                                                                                                                               CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][3]])
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            # Update vehicle ID
            if not passing_SX: vehicle_SX_ID += 1
            if not passing_DX: vehicle_DX_ID += 1
            
            passing_SX = False
            passing_DX = False
            
            # Wait for ESC key to stop
            if cv2.waitKey(30) != -1:
                capture.release()
                cv2.destroyAllWindows()
                break

            # Update data on the screen (almost) every second
            end_update = time.time()
            update_interval += end_update - start_update
            if update_interval >= 1:
                FPS = int(1 / round(end_update - start_update, 3))
                year, month, day, hh, mm, ss = Video.timer(self, int(year), int(month), int(day), int(hh), int(mm), int(ss))
                update_interval = 0

            # Project name
            cv2.rectangle(frame, (5, 775), (290, 805), (200, 200, 200), -1)

            cv2.putText(frame, 'Powered by SmarTraffic', (10, 797), cv2.FONT_HERSHEY_DUPLEX, 0.7, (10, 10, 10), 1)

            # Display live statistics on the screen 
            if self.stats:
                cv2.rectangle(frame, (5, 5), (160, 105), (200, 200, 200), -1)
                cv2.putText(frame, 'FPS: {}'.format(FPS), (8, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (10, 10, 10), 1)
                cv2.putText(frame, 'Coming: {}'.format(vehicle_count_SX), (8, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (10, 10, 10), 1)
                cv2.putText(frame, 'Leaving: {}'.format(vehicle_count_DX), (8, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, (10, 10, 10), 1)
                cv2.putText(frame, 'Total: {}'.format(vehicle_count_SX + vehicle_count_DX), (8, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (10, 10, 10), 1)
                cv2.rectangle(frame, (780, 775), (1075, 805), (200, 200, 200), -1)
                cv2.putText(frame, '{}/{}/{} - {}:{}:{}'.format(day, month, year, hh, mm, ss), (788, 797), cv2.FONT_HERSHEY_DUPLEX, 0.7, (10, 10, 10), 1)

            # Show the video (and layer)
            cv2.imshow(CAMERA_SETTINGS[source[6 : len(source) - 4]]["Title"], frame)
            # cv2.imshow('Left lane', mask_SX)
            # cv2.imshow('Right lane', mask_DX)

    def detector(self):

        # Prepare CSV as database
        with open('data.csv', 'w') as data:
            data_input = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_input.writerow(['VEHICLE_ID', 'AREA', 'DETECTION', 'CONFIDENCE', 'DIRECTION', 'DATE', 'TIME', 'STATUS'])
            
        while True:

            # Check if there are photo to analyse
            if len(os.listdir('.tmp')):

                # Prepare the base for the ticket to generate
                base = cv2.imread('img/ticket.png')

                # Make detection for each photo
                for photo in os.listdir('.tmp')[:]:
                    
                    img = cv2.imread('.tmp/{}'.format(photo))
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

                    ticket = base.copy()
                    
                    # One object detected in the photo
                    if len(indexes) == 1:

                        x, y, w, h = boxes[indexes.flatten()[0]]
                    
                        obj = str(self.classes[class_ids[indexes.flatten()[0]]]) + ' - '
                        confidence = str(round(confidences[indexes.flatten()[0]] * 100, 1)) + '%'
                        if photo[photo.index('_') + 1] == 'C':
                            direction = 'Coming'
                        else:
                            direction = 'Leaving'  
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        if obj[ : len(obj) - 3] not in self.vehicles:
                            if obj[ : len(obj) - 3] in self.other_object :
                                status = 'ATTENTION: {} on the road'.format(obj[ : len(obj) - 3])
                            else:
                                status = 'ATTENTION: object on the road'
                            status_color = (0, 43, 214)
                        else:
                            status = 'OK'
                            status_color = (0, 179, 69)

                    # Manage detection anomalies
                    else:
                        
                        # No object detected in the photo
                        if len(indexes) == 0:

                            obj = '-'
                            confidence = ''
                            if photo[photo.index('_') + 1] == 'C':
                                direction = 'Coming'
                            else:
                                direction = 'Leaving'
                            status = 'ERROR: no object detected'
                            status_color = (0, 179, 219)

                        # Multiple object detected in the photo
                        else:

                            for i in indexes.flatten():
                                x, y, w, h = boxes[i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            obj = '-'
                            confidence = ''
                            if photo[photo.index('_') + 1] == 'C':
                                direction = 'Coming'
                            else:
                                direction = 'Leaving'
                            status = 'ERROR: multiple object detected'
                            status_color = (0, 179, 219)

                    # Compile and generate the ticket 
                    ticket[17 : 381, 19 : 272] = cv2.resize(img, (253, 364))
                    cv2.putText(ticket, '{}'.format(photo[ : photo.index('_')]), (415, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1) 
                    cv2.putText(ticket, '{}'.format(CAMERA_SETTINGS[self.camera[6 : len(self.camera) - 4]]["Title"]), (354, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1)
                    cv2.putText(ticket, '{}{}'.format(obj, confidence), (417, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1)
                    cv2.putText(ticket, '{}'.format(direction), (408, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1)
                    cv2.putText(ticket, '{}/{}/{}'.format(photo[photo.index('_') + 2 : photo.index('_') + 4], 
                                                        photo[photo.index('_') + 4 : photo.index('_') + 6],
                                                        photo[photo.index('_') + 6 : photo.index('_') + 10]), (355, 203), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1)
                    cv2.putText(ticket, '{}:{}:{}'.format(photo[photo.index('_') + 10 : photo.index('_') + 12], 
                                                        photo[photo.index('_') + 12 : photo.index('_') + 14],
                                                        photo[photo.index('_') + 14 : photo.index('_') + 16]), (361, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 1)
                    cv2.putText(ticket, '{}'.format(status), (370, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                    # Save the ticket
                    cv2.imwrite('detections/ticket_{}.png'.format(len(os.listdir('detections')) + 1), ticket)

                    # Add record to CSV database
                    with open('data.csv', 'a') as data:
                        data_input = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        data_input.writerow([photo[ : photo.index('_')],
                                            CAMERA_SETTINGS[self.camera[6 : len(self.camera) - 4]]["Title"],
                                            obj[ : len(obj) - 3],
                                            confidence,
                                            direction, 
                                            '{}/{}/{}'.format(photo[photo.index('_') + 2 : photo.index('_') + 4], photo[photo.index('_') + 4 : photo.index('_') + 6], photo[photo.index('_') + 6 : photo.index('_') + 10]),
                                            '{}:{}:{}'.format(photo[photo.index('_') + 10 : photo.index('_') + 12], photo[photo.index('_') + 12 : photo.index('_') + 14], photo[photo.index('_') + 14 : photo.index('_') + 16]),
                                            status])
                    
                    # Delete original photo
                    os.remove('.tmp/{}'.format(photo))
                    
            else:
                time.sleep(3)

    def timer(self, year, month, day, h, m, s):

        # Update the timer on the camera screen

        if s < 59:
            s += 1
        else:
            s = 0
            if m < 59:
                m += 1
            else:
                m = 0
                if h < 23:
                    h += 1
                else:
                    h = 0
                    if day < calendar.monthrange(year, month)[1]:
                        day += 1
                    else:
                        day = 1
                        if month < 12:
                            month += 1
                        else:
                            month = 1
                            year += 1

        if s < 10: s = '0' + str(s)
        if m < 10: m = '0' + str(m)
        if h < 10: h = '0' + str(h)
        if day < 10: day = '0' + str(day)
        if month < 10: month = '0' + str(month)
        if year < 10: year = '0' + str(year)
        
        return year, month, day, h, m, s

if __name__ == '__main__':

    start('video/camera_1.mp4', 1, 1)
