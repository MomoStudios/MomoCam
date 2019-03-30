"""
Usage:

python3 demo/classify_capture_opencv.py \
    --model test_data/inception_v4_299_quant_edgetpu.tflite  \
    --label test_data/imagenet_labels.txt

"""
import argparse
import io
import time
import sys
import subprocess

import numpy as np

import cv2
from PIL import Image
from PIL import ImageDraw

from edgetpu.detection.engine import DetectionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = DetectionEngine(args.model)

    try:
        cap = cv2.VideoCapture(0)
        _, width, height, channels = engine.get_input_tensor_shape()
        print('test1')
        print('test2')
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_processing = True
        while True:
            ret, frame = cap.read()

            resized = cv2.resize(frame, (width, height))
            if not is_processing:
                is_processing = True
                input = np.frombuffer(resized, dtype=np.uint8)
                start_time = time.time()
                results = engine.ClassifyWithInputTensor(input, top_k=1)
                elapsed_time = time.time() - start_time
                if results:
                    confidence = results[0][1]
                    label = labels[results[0][0]]
                    print("Elapsed time: {:0.02f}".format(elapsed_time * 1000))
                cv2.putText(frame, label, (0, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "{:0.02f}".format(confidence), (0, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

            sys.stdout.write(frame.tostring())

    except Exception as e:
        print('thats no good!')
        print(e)
    except KeyboardInterrupt:
        print('Shutting down...')
    finally:
        cap.release()
        cv2.destroyAllWindows()

#Pass in the numpy array of the frame and the detection engine
#output is (boxes, frame) where
#boxes is a list of bounding boxes added
#frame is the stillframe with the boxes added
def analyze_frame(frame, boxes, engine):
    boxes = []
    img = Image.fromarray(frame)
    ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
        relative_coord=False, top_k=10)
    for obj in ans:
        print ('-----------------------------------------')
        if labels:
            print(labels[obj.label_id])
        print ('score = ', obj.score)
        box = obj.bounding_box.flatten().tolist()
        print ('box = ', box)
        # Draw a rectangle.
        if label_is_cat(obj.label_id):
            boxes.append(box)
    # draw_boxes_on_picture(boxes, img)
    # out = np.array(img)

#draws the specified boxes in red on the input picture
def draw_boxes_on_picture(boxes, img):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, outline='red')


def label_is_cat(label_id):
    return True # TODO

if __name__ == '__main__':
    main()
