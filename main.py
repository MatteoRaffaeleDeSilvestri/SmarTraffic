import os
import cv2
import csv
import sys
import json
import numpy
import webbrowser
import tkinter as tk
import multiprocessing
from tkinter import ttk
from copy import deepcopy
from time import time, sleep
from tkinter import filedialog
from tkinter import messagebox
from calendar import monthrange
from tkinter import font as tkFont

class Video:

    def __init__(self, ntt, clss, source, dp, sts):

        # Initialise video variables
        self.net = ntt
        self.classes = clss
        self.camera = source
        self.detection_point = dp
        self.stats = sts

        # Type of vehicle that can be seen on the road (from coco.names)
        self.vehicles = {'Bicycle', 'Car', 'Motorbike', 'Bus', 'Truck', 'Boat'}

        # Common things to be aware of during drive (other person and animals mostly)
        self.other_object = {'Person', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Bear', 'Bird'}

    def play(self):

        # Set mini logo from camera screen
        logo_mini = cv2.imread('resources/logo_mini.png')

        # Take video from source 
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
        clock = CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"][CAMERA_SETTINGS[source[6 : len(source) - 4]]["Date-time"].index('-') + 2 : ].split(':')
        year, month, day, hh, mm, ss = timer(int(date[2]),  int(date[1]), int(date[0]), int(clock[0]), int(clock[1]),int(clock[2]))

        # Background subtractor from camera
        object_detector = cv2.createBackgroundSubtractorMOG2(history=CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][0], varThreshold=CAMERA_SETTINGS[source[6 : len(source) - 4]]["BackgroundSubtractor"][1])

        while True:

            ret, frame = capture.read()

            if not ret:
                capture.release()
                cv2.destroyAllWindows()
                break
            
            # Keep a "clean" copy of the frame for detection (for ticket generation)
            original = deepcopy(frame)

            # Start the timer for update the info on the screen (update every second)
            start_update = time()

            # Define regions of interest (ROI)
            roi_SX = frame[CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][1],
                           CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][3]]
            roi_DX = frame[CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][1],
                           CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][3]]

            # Generate mask for ROI
            mask_SX = object_detector.apply(roi_SX)
            mask_DX = object_detector.apply(roi_DX)

            # Remove "noise" from the frame
            _, mask_SX = cv2.threshold(mask_SX, 254, 255, cv2.THRESH_BINARY)
            _, mask_DX = cv2.threshold(mask_DX, 254, 255, cv2.THRESH_BINARY)

            # Find contours for moving object
            contours_SX, _ = cv2.findContours(mask_SX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_DX, _ = cv2.findContours(mask_DX, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find contours ROI_SX
            for contours in contours_SX:

                # Check areas and ignore small elements
                if cv2.contourArea(contours) >= CAMERA_SETTINGS[source[6 : len(source) - 4]]["Surface"]:
                    
                    x, y, w, h = cv2.boundingRect(contours)

                    # Activate detection point
                    if CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] + y <= CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1] <= CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_SX"][0] + y + h:
                        passing_SX = True

            # Find contours ROI DX
            for contours in contours_DX:

                  # Check areas and ignore small elements
                if cv2.contourArea(contours) >= CAMERA_SETTINGS[source[6 : len(source) - 4]]["Surface"]:
                    
                    x, y, w, h = cv2.boundingRect(contours)
                    
                    # Activate detection point
                    if CAMERA_SETTINGS[source[6 :len(source) - 4]]["ROI_DX"][0] + y <= CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1] <= CAMERA_SETTINGS[source[6 : len(source) - 4]]["ROI_DX"][0] + y + h:
                        passing_DX = True

            # Draw detection point
            if self.detection_point:
                cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (0, 0, 255), 2)
            
            # Take a picture of left lane (ROI)
            if passing_SX:
                if '{}.png'.format(str(vehicle_SX_ID) + '_C')[ : '{}'.format(str(vehicle_SX_ID) + '_C').index('C') + 1] not in [photo[ : '{}'.format(str(vehicle_SX_ID) + '_C').index('C') + 1] for photo in os.listdir('.tmp')]:
                    vehicle_count_SX += 1
                    cv2.imwrite('.tmp/{}.png'.format(str(vehicle_SX_ID) + '_C' + '{}{}{}{}{}{}'.format(day, month, year, hh, mm, ss)), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][1],
                                                                                                                                               CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_SX"][3]])
                
                # Animate detection point 
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            # Take a picture of right lane (ROI)
            if passing_DX:
                if '{}.png'.format(str(vehicle_DX_ID) + '_L')[ : '{}'.format(str(vehicle_DX_ID) + '_L').index('L') + 1] not in [photo[ : '{}'.format(str(vehicle_SX_ID) + '_L').index('L') + 1] for photo in os.listdir('.tmp')]:
                    vehicle_count_DX += 1
                    cv2.imwrite('.tmp/{}.png'.format(str(vehicle_DX_ID) + '_L' + '{}{}{}{}{}{}'.format(day, month, year, hh, mm, ss)), original[CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][0] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][1],
                                                                                                                                               CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][2] : CAMERA_SETTINGS[source[6 : len(source) - 4]]["Area_DX"][3]])
                
                # Animate detection point 
                if self.detection_point:
                    cv2.line(frame, (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0] + ((CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2] - CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][0]) // 2), CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][1]),
                                    (CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][2], CAMERA_SETTINGS[source[6 : len(source) - 4]]["DetectionPoint"][3]), (255, 255, 255), 2)
            
            # Update vehicle ID
            if not passing_SX: vehicle_SX_ID += 1
            if not passing_DX: vehicle_DX_ID += 1
            
            passing_SX = False
            passing_DX = False
            
            # Wait for ESC key to stop
            if cv2.waitKey(30) == 27:
                capture.release()
                cv2.destroyAllWindows()
                break

            # Update data on the screen (almost every second)
            end_update = time()
            update_interval += end_update - start_update
            if update_interval >= 1:
                FPS = int(1 / round(end_update - start_update, 3))
                year, month, day, hh, mm, ss = timer(int(year), int(month), int(day), int(hh), int(mm), int(ss))
                update_interval = 0

            # Default camera info
            cv2.rectangle(frame, (5, 775), (1075, 805), (230, 230, 230), -1)
            cv2.putText(frame, 'Powered by', (10, 797), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, 'Press ESC to stop the video', (370, 797), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
            frame[777 : 804 , 145 : 244] = logo_mini
            cv2.putText(frame, '{}/{}/{} - {}:{}:{}'.format(day, month, year, hh, mm, ss), (788, 797), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # Display live statistics on the screen 
            if self.stats:
                cv2.rectangle(frame, (5, 5), (160, 105), (200, 200, 200), -1)
                cv2.putText(frame, 'FPS: {}'.format(FPS), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(frame, 'Entering: {}'.format(vehicle_count_SX), (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(frame, 'Leaving: {}'.format(vehicle_count_DX), (8, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(frame, 'Total: {}'.format(vehicle_count_SX + vehicle_count_DX), (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # Show the video
            cv2.imshow(CAMERA_SETTINGS[source[6 : len(source) - 4]]["Title"], frame)
            
            # Uncomment code below to see the mask on the video  
            # cv2.imshow('Left lane', mask_SX)
            # cv2.imshow('Right lane', mask_DX)

    def detector(self, mode, tkt, tkt_folder, CSV, CSV_folder):

        # Prepare CSV file as database
        if CSV:
            with open('{}/data.csv'.format(CSV_folder), 'w') as data:
                data_input = csv.writer(data, delimiter=',')
                data_input.writerow(['VEHICLE_ID', 'AREA', 'DETECTION', 'CONFIDENCE', 'DIRECTION', 'DATE', 'TIME', 'STATUS'])
        
        # Ticket counter
        ticket_counter = 0

        while True:

            # Check if there are photos to analyse
            if len(os.listdir('.tmp')):

                # Prepare the base for the ticket to generate
                base = cv2.imread('resources/ticket.png')

                # Make detection for each photo
                for photo in os.listdir('.tmp')[:]:
                    
                    # Take photo size
                    img = cv2.imread('.tmp/{}'.format(photo))
                    height, width, _ = img.shape

                    # Prepare for object recognition
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

                            # Set the minimum level of confidence at 50%
                            if confidence > 0.5:
                        
                                # Get object size
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

                    # Get ticket image base to fill with informations
                    ticket = base.copy()
                    
                    # STANDARD CASE: one object detected in the photo
                    if len(indexes) == 1:

                        x, y, w, h = boxes[indexes.flatten()[0]]

                        obj = '{} - '.format(str(self.classes[class_ids[indexes.flatten()[0]]]))
                        confidence = '{}%'.format(str(round(confidences[indexes.flatten()[0]] * 100, 1)))
                        if photo[photo.index('_') + 1] == 'C':
                            direction = 'Entering'
                        else:
                            direction = 'Leaving' 
                        
                        if obj[ : len(obj) - 3] not in self.vehicles:
                            if obj[ : len(obj) - 3] in self.other_object :
                                status = 'ATTENTION: {} on the road'.format(obj[ : len(obj) - 3])
                            else:
                                status = 'ATTENTION: object on the road'
                            status_color = (0, 43, 214)

                        else:

                            status = 'OK'
                            status_color = (0, 179, 69)

                        cv2.rectangle(img, (x, y), (x + w, y + h), status_color, 2)

                    # OTHER CASES: manage detection anomalies
                    else:
                        
                        # No object detected in the photo
                        if len(indexes) == 0:

                            obj = '-'
                            confidence = ''
                            if photo[photo.index('_') + 1] == 'C':
                                direction = 'Entering'
                            else:
                                direction = 'Leaving'
                            status = 'ERROR: no object detected'
                            status_color = (0, 179, 219)

                        # Multiple object detected in the photo
                        else:

                            obj = '-'
                            confidence = ''
                            if photo[photo.index('_') + 1] == 'C':
                                direction = 'Entering'
                            else:
                                direction = 'Leaving'
                            status = 'ERROR: multiple object detected'
                            status_color = (0, 179, 219)

                            for i in indexes.flatten():
                                x, y, w, h = boxes[i]
                                cv2.rectangle(img, (x, y), (x + w, y + h), status_color, 2)
                    
                    # Compile and generate the ticket 
                    if tkt:

                        ticket[17 : 381, 19 : 272] = cv2.resize(img, (253, 364))
                        cv2.putText(ticket, '{}'.format(photo[ : photo.index('_')]), (415, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1) 
                        cv2.putText(ticket, '{}'.format(CAMERA_SETTINGS[self.camera[6 : len(self.camera) - 4]]["Title"]), (354, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                        cv2.putText(ticket, '{}{}'.format(obj, confidence), (417, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                        cv2.putText(ticket, '{}'.format(direction), (408, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                        cv2.putText(ticket, '{}/{}/{}'.format(photo[photo.index('_') + 2 : photo.index('_') + 4], 
                                                            photo[photo.index('_') + 4 : photo.index('_') + 6],
                                                            photo[photo.index('_') + 6 : photo.index('_') + 10]), (355, 203), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                        cv2.putText(ticket, '{}:{}:{}'.format(photo[photo.index('_') + 10 : photo.index('_') + 12], 
                                                            photo[photo.index('_') + 12 : photo.index('_') + 14],
                                                            photo[photo.index('_') + 14 : photo.index('_') + 16]), (361, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                        cv2.putText(ticket, '{}'.format(status), (370, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                        # Update ticket counter and save ticket
                        ticket_counter += 1
                        cv2.imwrite('{}/ticket_{}.png'.format(tkt_folder.split(' ')[0], ticket_counter), ticket)

                    # Add record to CSV file
                    if CSV:
                        with open('{}/data.csv'.format(CSV_folder), 'a') as data:
                            data_input = csv.writer(data, delimiter=',')
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
                
                # Stop for "non-linux OS"
                if mode: break
                    
            else:

                sleep(1)

class GUI:

    def __init__(self):

        # Prepare camera for selection in GUI
        self.cameras = dict()
        cam_number = 0
        for camera in CAMERA_SETTINGS:
            cam_number += 1
            self.cameras.setdefault('Camera {} - {}'.format(cam_number, CAMERA_SETTINGS[camera]["Title"]), 'camera_{}.mp4'.format(cam_number))

        # Prepare the object recognition system (YoloV4)
        self.net = cv2.dnn.readNet('yolo/yolov4.weights', 'yolo/yolov4.cfg')
        
        self.classes = list()
        with open('yolo/coco.names', 'r') as f:
            self.classes = f.read().splitlines()

        # Set window propriety
        self.root = tk.Tk()
        self.root.title('[DEMO] SmarTraffic')
        self.root.resizable(False, False)

        # Set font style for GUI
        lato14 = tkFont.Font(family='Lato', size=14)
        self.lato13 = tkFont.Font(family='Lato', size=13)
        self.lato12italic = tkFont.Font(family='Lato', size=12, slant='italic')
        lato12 = tkFont.Font(family='Lato', size=12)
        lato11 = tkFont.Font(family='Lato', size=11)
        lato10italic = tkFont.Font(family='Lato', size=10, slant='italic')
        
        # Show logo
        logo = tk.Canvas(self.root, width=445, height=120)
        logo.grid(row=0, column=0, columnspan=2, padx=50, pady=15)
        logo_img = tk.PhotoImage(file='resources/logo.png')
        logo.create_image(0, 0, anchor='nw', image=logo_img)

        # Welcome message
        tk.Label(self.root, font=lato14, fg='#242424', text='Welcome to SmarTraffic!\nThis demo aims to show how the program works.\nFollow the steps below and see the result').grid(row=1, column=0, columnspan=2, padx=20, pady=15)

        # Horizontal separator
        ttk.Separator(self.root, orient='horizontal').grid(row=2, column=0, columnspan=2, padx=10, sticky='ew')

        # STEP 1
        tk.Label(self.root, font=lato12, fg='#242424', text='STEP 1\nChoose a camera from the menù below and the information you want\nto display on the camera screen').grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

        # Initialise and show dropdown menù
        sources = [video for video in self.cameras.keys()]

        variable = tk.StringVar(self.root)
        variable.set(sources[0])
        dropdown_menu = tk.OptionMenu(self.root, variable, *sources)
        dropdown_menu.config(font=self.lato13, fg='#242424', width=28)
        
        dropdown_menu.grid(row=4, column=0, rowspan=2, padx=20, sticky='ew')

        other_options = self.root.nametowidget(dropdown_menu.menuname)
        other_options.config(font=self.lato13, fg='#242424')

        # Show detection point
        dp = tk.IntVar()
        dp.set(0)
        tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Show detection point', variable=dp, width=28).grid(row=4, column=1, padx=25, sticky='w')

        # Show live statistics
        sts = tk.IntVar()
        sts.set(0)
        tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Show live statistics   ', variable=sts, width=28).grid(row=5, column=1, padx=22, sticky='w')

        # Horizontal separator
        ttk.Separator(self.root, orient='horizontal').grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky='ew')

        # STEP 2
        tk.Label(self.root, font=lato12, fg='#242424', text='STEP 2\nChoose if to generate or not the tickets and the CSV data file').grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky='ew')

        # Save ticket
        tkt = tk.IntVar()
        tkt.set(0)
        tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Save tickets', variable=tkt, command=lambda: GUI.update_folder(self, tkt, 0)).grid(row=8, column=0, padx=25, sticky='ew')
        
        tk.Button(self.root, font=self.lato13, fg='#242424', text='Choose a destination folder', command=lambda: GUI.dest_directory(self, 0), width=28, state='disabled').grid(row=9, column=0, padx=30, pady=10, sticky='ew')

        self.tkt_folder = tk.Label(self.root, font=self.lato12italic, fg='#a0a0a0', text='Current folder:\n - ',  width=40)
        self.tkt_folder.grid(row=10, column=0, padx=25, sticky='ew')

        # Export data as CSV file
        csv_file = tk.IntVar()
        csv_file.set(0)
        tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Export data as CSV file', variable=csv_file, command=lambda: GUI.update_folder(self, csv_file, 1)).grid(row=8, column=1, padx=25, sticky='ew')

        tk.Button(self.root, font=self.lato13, fg='#242424', text='Choose a destination folder', command=lambda: GUI.dest_directory(self, 1), width=28, state='disabled').grid(row=9, column=1, padx=30, pady=10, sticky='ew')
        
        self.csv_folder = tk.Label(self.root, font=self.lato12italic, fg='#a0a0a0', text='Current folder:\n - ', width=40)
        self.csv_folder.grid(row=10, column=1, padx=25, sticky='ew')
        
        # Horizontal separator
        ttk.Separator(self.root, orient='horizontal').grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky='ew')

        # STEP 3
        tk.Label(self.root, font=lato12, fg='#242424', text='STEP 3\nPress \"Play\" and see how the program work').grid(row=12, column=0, columnspan=2, padx=5, pady=10, sticky='ew')

        # Play button
        tk.Button(self.root, font=self.lato13, bg='#53c918', fg='#242424', activebackground='#80ff40', text='Play', command=lambda: GUI.play_update(self, variable, dp, sts, tkt, self.tkt_folder['text'], csv_file, self.csv_folder['text']), width=8, state='normal').grid(row=13, column=0, columnspan=2, pady=10,)

        # Horizontal separator
        ttk.Separator(self.root, orient='horizontal').grid(row=14, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

        # Tutorial button
        tk.Label(self.root, font=lato12, fg='#242424', text='Need some help?').grid(row=15, column=0, columnspan=2)
        tk.Button(self.root, font=self.lato12italic, fg='#242424', activeforeground='#001263', text='Check out the documentation', command=lambda: webbrowser.open('https://github.com/MatteoRaffaeleDeSilvestri/SmarTraffic', new=0, autoraise=True)).grid(row=16, column=0, columnspan=2, padx=10, pady=5)

        # Author
        tk.Label(self.root, font=lato10italic, fg='#a0a0a0', text='MatteoRaffaeleDeSilvestri').grid(row=17, column=0, columnspan=2, sticky='e')

        # Start main loop (GUI)
        self.root.mainloop()
    
    def update_folder(self, var, col):

        # Update the button after selecting/deselecting the associated option
        new_btn = tk.Button(self.root, font=self.lato13, fg='#242424', text='Choose a destination folder', command=lambda: GUI.dest_directory(self, col))
        
        if var.get():
            new_btn['state'] = 'normal'
            color = '#242424'
        else:
            color = '#a0a0a0'
            new_btn['state'] = 'disabled'

        new_btn.grid(row=9, column=col, padx=30, pady=10, sticky='ew')

        if col:
            self.csv_folder = tk.Label(self.root, font=self.lato12italic, fg=color, text='Current folder:\n - ', width=40)
            self.csv_folder.grid(row=10, column=col, padx=25, sticky='ew')
        else:
            self.tkt_folder = tk.Label(self.root, font=self.lato12italic, fg=color, text='Current folder:\n - ', width=40)
            self.tkt_folder.grid(row=10, column=col, padx=25, sticky='ew')
        
        # Call directory choice function automatically
        if color == '#242424':
            GUI.dest_directory(self, col)

    def dest_directory(self, col):

        # Reset previous folder choice
        if col:
            self.csv_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n - ', width=40)
            self.csv_folder.grid(row=10, column=col, padx=25, sticky='ew')
        else:
            self.tkt_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n - ', width=40)
            self.tkt_folder.grid(row=10, column=col, padx=25, sticky='ew')

        # Choose the destination folder for generated file
        folder = filedialog.askdirectory(mustexist=True)
        
        if type(folder) == str and os.path.isdir(str(folder)):

            if col:
                self.csv_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n{}'.format(folder), width=40)

                self.csv_folder.grid(row=10, column=col, padx=25, sticky='ew')
            else:
                self.tkt_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n{}'.format(folder), width=40)
                self.tkt_folder.grid(row=10, column=col, padx=25, sticky='ew')
        
        else:

            # Ask what to do to the user
            warning = messagebox.askquestion('Invalid folder', 'The selected folder is not valid!\n\nIf you want to save the file you need to select a folder.\nDo you want to save the file?')
            if warning == 'yes':
                
                # Repeat the folder selection procedure
                GUI.dest_directory(self, col)
                
            else:

                # "Reset" folder
                if col:
                    self.csv_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n - ', width=40)
                    self.csv_folder.grid(row=10, column=col, padx=25, sticky='ew')
                else:
                    self.tkt_folder = tk.Label(self.root, font=self.lato12italic, fg='#242424', text='Current folder:\n - ', width=40)
                    self.tkt_folder.grid(row=10, column=col, padx=25, sticky='ew')

    def play_update(self, variable, dp, sts, tkt, tkt_folder, csv_file, csv_file_folder):
        
        # Check ticket and CSV folder anomalies
        if tkt.get() and not os.path.isdir('/{}'.format(tkt_folder[tkt_folder.index(':') + 3 : ])):

            # Notify the error to the user
            tkt_answare = messagebox.askyesno('Invalid folder', 'No folder selected.\n\nTickets will not be saved.\nDo you want to continue?')
            if not tkt_answare:
                return

            # Update ticket section (GUI)
            tkt = tk.IntVar()
            tkt.set(0)
            tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Save tickets', variable=tkt, command=lambda: GUI.update_folder(self, tkt, 0)).grid(row=8, column=0, padx=25, sticky='ew')
            GUI.update_folder(self, tkt, 0)

        if csv_file.get() and not os.path.isdir('/{}'.format(csv_file_folder[csv_file_folder.index(':') + 3 : ])):
        
            # Notify the error to the user
            csv_answare = messagebox.askyesno('Invalid folder', 'No folder selected.\n\nCSV file will not be saved.\nDo you want to continue?')
            if not csv_answare:
                return

            # Update CSV section (GUI)
            csv_file = tk.IntVar()
            csv_file.set(0)
            tk.Checkbutton(self.root, font=self.lato13, fg='#242424', text='Export data as CSV file', variable=csv_file, command=lambda: GUI.update_folder(self, csv_file, 1)).grid(row=8, column=1, padx=25, sticky='ew')
            GUI.update_folder(self, csv_file, 1)

        # "Lock" play button
        tk.Button(self.root, font=self.lato13, bg='#b3fc8d', fg='#242424', text='Playing', width=8, state='disabled').grid(row=13, column=0, columnspan=2)
        self.root.update()
        
        # Run the program
        GUI.run(self, 'video/{}'.format(self.cameras[variable.get()]), dp.get(), sts.get(), tkt.get(), '/{}'.format(tkt_folder[tkt_folder.index(':') + 3 : ]), csv_file.get(), '/{}'.format(csv_file_folder[csv_file_folder.index(':') + 3 : ]))
        
        # "Unlock" play button
        tk.Button(self.root, font=self.lato13, bg='#53c918', fg='#242424', activebackground='#80ff40', text='Play', command=lambda: GUI.play_update(self, variable, dp, sts, tkt, self.tkt_folder['text'], csv_file, self.csv_folder['text']),  width=8, state='normal').grid(row=13, column=0, columnspan=2)
        self.root.update()

    def run(self, source, dp, sts, ticket, ticket_folder, CSV, CSV_folder):

        # Clean .tmp folder from old photos (if present)
        for photo in os.listdir('.tmp')[:]:
            os.remove('.tmp/{}'.format(photo))
        
        video = Video(self.net, self.classes, source, dp, sts)

        # Use multiprocessing to simulate real-time operations in linux OS 
        if sys.platform == 'linux':
        
            # Initialize process
            play = multiprocessing.Process(target=video.play, args=[])
            detect = multiprocessing.Process(target=video.detector, args=[0, ticket, ticket_folder, CSV, CSV_folder])

            # Starting multiprocessing procedure
            play.start()
            detect.start()

            timeout = 1
            while True:

                # Keep running until the video isn't ended (or the user stops it)
                if play.exitcode == 0:
                    timeout = 0.1

                    # If the video is ended complete detection on photos on .tmp folder (if present)
                    if not len(os.listdir('.tmp')):
                        os.kill(detect.pid, 9) 
                        break

                sleep(timeout)
        
        # Manage detection procedure for "non-linux OS"
        else:
            
            # Play video
            video.play()

            # Make object recognition (AFTER playing video)
            video.detector(1, ticket, ticket_folder, CSV, CSV_folder)

def timer(year, month, day, h, m, s):

    # Give the timer a proper format for visualization
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
                if day < monthrange(year, month)[1]:
                    day += 1
                else:
                    day = 1
                    if month < 12:
                        month += 1
                    else:
                        month = 1
                        year += 1
    
    # "Normalise" values before returning
    if s < 10: s = '0' + str(s)
    if m < 10: m = '0' + str(m)
    if h < 10: h = '0' + str(h)
    if day < 10: day = '0' + str(day)
    if month < 10: month = '0' + str(month)
    if year < 10: year = '0' + str(year)
    
    return year, month, day, h, m, s
    
if __name__ == '__main__':

    # Import camera settings
    with open('CAMERA_SETTINGS.json', 'r') as f:
        CAMERA_SETTINGS = json.load(f)

    # Create '.tmp' folder (if not exsist)
    if not os.path.isdir('.tmp'): os.mkdir('.tmp')

    # Start SmarTraffic
    GUI()
