import cv2
import tkinter as tk
import main
import os, webbrowser
    
# def Openfolder(x):
#     print(x)
#     webbrowser.open('analysed')

class GUI:

    cameras = {
            'Camera 1 - Via Fondi-Sperlonga': 'camera_1.mp4',
            'Camera 2 - Via Appia Lato Itri': 'camera_2.mp4',
            'Camera 3 - SR637': 'camera_3.mp4',
            'Camera 4 - SS7': 'camera_4.mp4',
    }

    def __init__(self):

        # Set window propriety
        self.root = tk.Tk()
        self.root.title('Smart Traffic')
        self.root.resizable(False, False)


        # Show logo
        logo = tk.Canvas(self.root, width=452, height=114)
        logo.grid(row=0, padx=20, pady=30)
        logo_img = tk.PhotoImage(file='logo.png')
        logo.create_image(0, 0, anchor='nw', image=logo_img)

        lbl = tk.Label(self.root, font='Lato', text='Benvenuto in SmarTraffic.\nSeleziona una delle telecamere presenti dal menù in basso e premi Play.', padx=20, pady=10)
        lbl.grid(columnspan=2, row=1)
 
        # Initialise and show dropdown menù
        sources = [video for video in self.cameras.keys()]

        variable = tk.StringVar(self.root)
        variable.set(sources[0])
        dropdown_menu = tk.OptionMenu(self.root, variable, *sources).grid(row=2, pady=10)

        dp = tk.IntVar()
        box = tk.Checkbutton(self.root, text='Mostra punti di rilevamento', variable=dp).grid(row=3, pady=10)

        sts = tk.IntVar()
        box = tk.Checkbutton(self.root, text='Mostra statistiche live', variable=sts).grid(row=4, pady=10)

        self.play_btn = tk.Button(self.root, text='Play', command=lambda: GUI.play_update(self, variable, dp, sts)).grid(row=5, pady=10)

        self.root.mainloop()

    def play_update(self, variable, dp, sts):

        # Start video detection
        self.play_btn = tk.Button(self.root, text='Play', state='disabled').grid(row=5, pady=10)
        self.root.update()

        main.run('video/{}'.format(self.cameras[variable.get()]), dp.get(), sts.get())
        
        self.play_btn = tk.Button(self.root, text='Play', command=lambda: GUI.play_update(self, variable, dp, sts)).grid(row=5, pady=10)
        self.root.update()

if __name__ == '__main__':
    
    GUI()