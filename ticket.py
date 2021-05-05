import cv2
import numpy as np


ticket = cv2.imread('Ticket.png')
auto = cv2.imread('analysed/1.png')
img_3 = ticket.copy()
img_3[27 : 427, 120 : 370] = auto[0 : 400 , 0 : 250]
cv2.imshow('Ticket', img_3)
cv2.waitKey(0)

        