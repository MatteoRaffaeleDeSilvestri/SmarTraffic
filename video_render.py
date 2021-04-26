import cv2
import os
from os.path import isfile, join

def video_render(pathIn, pathOut, FPS, time):
    frame_array = list()
    files = [frame for frame in os.listdir(pathIn) if isfile(join(pathIn, frame))]
    
    for i in range(len(files)):
        filename = pathIn + files[i]

        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)

        for k in range(time):
            frame_array.append(img)
    
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)

    for i in range(0, len(frame_array), time):
        out.write(frame_array[i])
    
    out.release()

if __name__ == '__main__':
    directory = 'render'
    pathIn = directory + '/'
    pathOut = pathIn + 'out.mp4'
    FPS = 1
    time = 1
    video_render(pathIn, pathOut, FPS, time)