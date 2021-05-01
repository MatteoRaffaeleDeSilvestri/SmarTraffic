import cv2
import numpy as np
from os.path import isfile
from time import time

# Detect camera from GUI
camera = 'video/drone.mp4'

cap = cv2.VideoCapture(camera)

# Object detection from camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=180) # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS

# Cam variables
passing_on = False
car_id = 0
count = 0
data_update = 0
FPS = '-'

while True:

    _, frame = cap.read()

    if frame is None:
        print("Video ended")
        break
    
    start = time()

    frame = cv2.resize(frame, (1280, 720))
    original = cv2.resize(frame, (1280, 720))
    
    # Define region of interest (ROI)
    roi = frame[150 : 710, 390 : 670] # <--- THIS VALUE IS DIFFERENT DEPENDING ON THE CAMERAS

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

        if 1000 < area < 100000: # <--- THIS VALUE CAN BE DIFFERENT DEPENDING THE CAMERAS
            
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) 
            # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Animate detection point
            if 600 in range(150 + y, 150 + y + h):
                passing_on = True
      
    if passing_on:
        if not isfile('detections/{}.png'.format(car_id)):
            cv2.imwrite('detections/{}.png'.format(car_id), original[150 : 710, 400 : 700])
            count += 1
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
    data_update += end - start
    if data_update >= 1:
        FPS = int(1 / round(end - start, 3))
        data_update = 0
    
    cv2.putText(frame, 'FPS: {}'.format(FPS), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    cv2.putText(frame, 'Vehicles: {}'.format(count), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    # Show the video (and layer)
    cv2.imshow(camera[6 : len(camera) - 4], frame)
    # cv2.imshow('ROI {}'.format(camera[6 : len(camera) - 4]), mask)

cap.release()
cv2.destroyAllWindows()
