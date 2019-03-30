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
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'bgr24',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        'output.mp4' ] 
#        'rtmp://live-jfk.twitch.tv/app/' + os.environ['twitch'] ]

print(command)
proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out, err = proc.communicate()
    print(out)
    print(err)

    proc.stdin.write(frame.tostring())

cap.release()
proc.stdin.close()
proc.stderr.close()
proc.wait()