'''capture.py'''
import cv, sys
cap = cv.CaptureFromCAM(0)                    # 0 is for /dev/video0
while True :
    if not cv.GrabFrame(cap) : break
    frame = cv.RetrieveFrame(cap)
    sys.stdout.write( frame.tostring() )