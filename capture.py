'''capture.py'''
import cv2, sys
cap = cv2.VideoCapture(0)
while True :
    ret, frame = cap.read()
    sys.stdout.buffer.write( frame.tostring() )
