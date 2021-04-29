import cv2
import numpy as np
import os

# Detect camera from GUI
camera = 'video/drone.mp4'

cap = cv2.VideoCapture(camera)

# Object detection from camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS

while True:

    _, frame = cap.read()

    if frame is None:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Draw detection line on the main frame
    cv2.line(frame, (455, 600), (640, 600), (0, 0, 255), 2)
    # cv2.line(frame, (520, 390), (650, 390), (255, 0, 0), 1)
    # cv2.line(frame, (300, 600), (390, 680), (0, 0, 255), 2)
    # cv2.line(frame, (668, 400), (910, 400), (0, 0, 255), 2)

    # Define region of interest (ROI)
    roi = frame[250 : 710, 390 : 670] # <--- THIS VALUE IS DIFFERENT DEPENDING ON THE CAMERAS
    
    # Generate mask for object
    mask = object_detector.apply(roi)

    # Remove shadow and other noise from the image
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # Find contours for moving object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    for cnt in contours:

        # Calculate areas and remove small elements
        area = cv2.contourArea(cnt)

        if area >= 1000: # <--- THIS VALUE CAN BE DIFFERENT DEPENDING THE CAMERAS

            cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 1) 
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Animate detection point
            if 600 in range(x + 470, y + 470):
                cv2.line(frame, (455, 600), (640, 600), (255, 255, 255), 2)

            # if y + h == 470 and not os.path.isfile('photos/Detection{}.png'.format(car_id - 1)):
            #         cv2.imwrite('photos/Detection{}.png'.format(car_id), frame[250 : 720, 390 : 670])
            #         car_id += 1

    # Show the video (and layer)
    cv2.imshow(camera[6 : len(camera) - 4], frame)
    cv2.imshow('ROI {}'.format(camera[6 : len(camera) - 4]), mask)
    
    # Wait ESC key to stop the video
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
