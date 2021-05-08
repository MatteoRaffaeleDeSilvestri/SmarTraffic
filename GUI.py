import cv2
import tkinter as tk
from PIL import ImageTk, Image

class GUI:
    
    def __init__(self):

        # Capture from camera
        self.cap = cv2.VideoCapture('video/camera_2.mp4')
        
        root = tk.Tk()
        root.geometry('1500x900')
        root.title('Smart Traffic')
        root.resizable(False, False)
        
        label1 = tk.Label(root, text='Use an image as logo')
        label1.grid(column=0, row=0, padx=10, pady=30)

        player = tk.Frame(root, bg="white")
        player.grid(column=1, row=0)
        self.player_frame = tk.Label(player)
        self.player_frame.grid(column=1, row=0)

        
        root.mainloop()
        video_stream()

    def video_stream(self):
        _, frame = self.cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        player_frame.imgtk = imgtk
        player_frame.configure(image=imgtk)
        player_frame.after(1, video_stream)

if __name__ == '__main__':
    x = GUI()