import tkinter as tk
from PIL import ImageTk, Image
import cv2


root = tk.Tk()
# Create a frame
app = tk.Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = tk.Label(app)
lmain.grid()

# Capture from camera
cap = cv2.VideoCapture('video/drone.mp4')

# function for video streaming
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

video_stream()
root.mainloop()