import cv2
import tkinter as tk
import main
import os, webbrowser
    
def Openfolder():
    webbrowser.open('analysed')

cameras = {
    'Camera 1': 'camera_1.mp4',
    'Camera 2': 'camera_2.mp4',
    'Camera 3': 'camera_3.mp4',
    'Camera 4': 'camera_4.mp4',
}

root = tk.Tk()
root.title('Smart Traffic')
root.resizable(False, False)

logo = tk.Canvas(root, width=461, height=120)
logo.grid(columnspan=1, row=0, padx=40, pady=30)
logo_img = tk.PhotoImage(file='logo.png')
logo.create_image(0, 0, anchor='nw', image=logo_img)

sources = [video for video in cameras.keys()]
for video in sources:
    sources.remove
    video = video[:len(video) - 4].replace('_', ' ')

variable = tk.StringVar(root)
variable.set(sources[0])
dropdown_menu = tk.OptionMenu(root, variable, *sources)
dropdown_menu.grid(column=0, row=1, padx=10)

btn = tk.Button(root, text='Play', command=Openfolder) # command=lambda: main.run('video/camera_1.mp4'))
btn.grid(column=0, row=2, pady=40)

root.mainloop()
