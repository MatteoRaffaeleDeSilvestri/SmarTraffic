import tkinter as tk

class GUI:
    
    def __init__(self):
        
        root = tk.Tk()
        root.geometry('1500x900')
        root.title('Traffic Eye')
        root.mainloop()

if __name__ == '__main__':
    
    x = GUI()