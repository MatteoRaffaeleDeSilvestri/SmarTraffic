import cv2
import numpy as np
import glob
 
img_array = []

for filename in glob.glob('render/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()