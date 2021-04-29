import cv2
import numpy as np

# Detect camera from GUI
camera = 'video/camera_01.mp4'

cap = cv2.VideoCapture(camera)

# Object detection from camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS

while True:

    _, frame = cap.read()

    if frame is None:
        break

    # Draw detection line on the main frame
    cv2.line(frame, (5, 570), (560, 570), (255, 0, 0), 2)
    cv2.putText(frame, 'DETECTION POINT', (5, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.line(frame, (668, 400), (910, 400), (0, 0, 255), 2)
    cv2.putText(frame, 'DETECTION POINT', (668, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Define region of interest (ROI)
    roi = frame[380 : 570, 0 : 600] # <--- THIS VALUE IS DIFFERENT DEPENDING ON THE CAMERAS
    
    # Generate mask for object
    mask = object_detector.apply(roi)

    # Findt contours for vehicle
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    for cnt in contours:

        # Calculate areas and remove small elements
        area = cv2.contourArea(cnt)
        
        if area >= 500: # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1) 
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi, (x, y), (x + w, y + y), (0, 255, 0), 1)
    
    # Show the video (and layer)
    cv2.imshow(camera[6:len(camera) - 4], frame)
    cv2.imshow('ROI {}'.format(camera[6:len(camera) - 4]), mask)
    
    # Wait ESC key to stop the video
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()