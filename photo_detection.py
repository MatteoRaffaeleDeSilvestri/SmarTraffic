import cv2
import numpy as np

video = cv2.VideoCapture('video/camera_01.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=90) # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS

while True:

    _, frame = video.read()
    roi = frame[350: 720, 0: 600]
    grey_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    median_frame = cv2.medianBlur(grey_frame, 5)
    expand_frame = cv2.dilate(median_frame, np.ones((5, 5)))

    # roi = frame[350: 720, 0: 600]

    mask = object_detector.apply(roi)

    contours, _ = cv2.findContours(expand_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    for cnt in contours:

        # Calculate areas and remove small elements
        area = cv2.contourArea(cnt)
        
        # if area >= 500: # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS
        cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) 

    # cv2.imshow("Webcam", frame)
    cv2.imshow("Media", expand_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
