# This script copies the video frame by frame
import cv2
import subprocess as sp
import os

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width, ch = frame.shape

ffmpeg = 'ffmpeg'
dimension = '{}x{}'.format(width, height)
f_format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(cv2.CAP_PROP_FPS))

command = [ffmpeg,
        '-y',
        '-f', 'rawvideo',
        '-vcodec','libx264',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        '-preset', 'ultrafast',
        '-r', fps,
        'bufsize', '512k'
        '-i', '-',
        '-an',
        '-b:v', '2500k',
        '-f', 'flv',
        'rtmp://live-jfk.twitch.tv/app/' + os.environ['twitch'] ]

print(command)
proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    for line in iter(proc.stdout.readline, ''):  # replace '' with b'' for Python 3
        sys.stdout.write(line)
    proc.stdin.write(frame.tostring())

cap.release()
proc.stdin.close()
proc.stderr.close()
proc.wait()