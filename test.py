import cv2

# Detect camera from GUI.
camera = 'video/camera_02.mp4'

cap = cv2.VideoCapture(camera)

# Object detection from camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=150) # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS

while True:

    ret, frame = cap.read()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # height, width, _ = frame.shape
    # print(height, width)
    
    # Define region of interest (ROI)
    roi = frame[350: 720, 0: 1280 ] # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS
    
    # Generate mask for object
    mask = object_detector.apply(roi)
    
    # Findt contours for vehicle
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    for cnt in contours:

        # Calculate areas and remove small elements
        area = cv2.contourArea(cnt)
        
        if area >= 1000: # <--- THIS VALUE CAN BE DIFFERENT DEPENDING ON THE CAMERAS
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 1) 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + y), (0, 255, 0), 2)
    
    # Show the video (and layer) 
    cv2.imshow(camera[6:len(camera) - 4], frame)
    cv2.imshow('Mask {}'.format(camera[6:len(camera) - 4]), mask)
    # cv2.imshow('ROI {}'.format(camera[6:len(camera) - 4]), roi)

    # Wait ESC key to stop the video
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()