import cv2
import tkinter as tk
import main
import os, webbrowser
    
# def Openfolder(x):
#     print(x)
#     webbrowser.open('analysed')

class GUI:

    def __init__(self):

        # Set window propriety
        root = tk.Tk()
        root.title('Smart Traffic')
        root.resizable(False, False)

        # Show logo
        logo = tk.Canvas(root, width=452, height=114)
        logo.grid(columnspan=1, row=0, padx=40, pady=30)
        logo_img = tk.PhotoImage(file='logo.png')
        logo.create_image(0, 0, anchor='nw', image=logo_img)

        cameras = {
            'Camera 1 - Via Fondi-Sperlonga': 'camera_1.mp4',
            'Camera 2 - Via Appia Lato Itri': 'camera_2.mp4',
            'Camera 3 - SR637': 'camera_3.mp4',
            'Camera 4 - SS7': 'camera_4.mp4',
        }

        sources = [video for video in cameras.keys()]

        variable = tk.StringVar(root)
        variable.set(sources[0])
        dropdown_menu = tk.OptionMenu(root, variable, *sources).grid(column=0, row=1, padx=10)

        dp = tk.IntVar()
        box = tk.Checkbutton(root, text='Enable detection point', variable=dp).grid(column=0, row=2)

        play_btn = tk.Button(root, text='Play', command=lambda: main.run('video/{}'.format(cameras[variable.get()]), dp.get())).grid(column=0, row=3, pady=40)
        show_btn = tk.Button(root, text='Show detections', state='disabled').grid(column=0, row=4, pady=40)

        root.mainloop()

if __name__ == '__main__':
    
    GUI()