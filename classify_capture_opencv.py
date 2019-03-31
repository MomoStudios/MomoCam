"""
Usage:

python3 demo/classify_capture_opencv.py \
    --model test_data/inception_v4_299_quant_edgetpu.tflite  \
    --label test_data/imagenet_labels.txt \
    [--synchronous] [--local]

"""
import argparse
import io
import time
import sys
import multiprocessing

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
    parser.add_argument(
      '--synchronous', help='Use to do anlysis synchronously.',
      required=False, action="store_true", default=False)
    parser.add_argument(
      '--local', help='send output to local window, instead of twitch',
      required=False, action="store_true", default=False)
    args = parser.parse_args()

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = DetectionEngine(args.model)

    try:
        cap = cv2.VideoCapture(0)
        _, width, height, channels = engine.get_input_tensor_shape()
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_processing = True
        child = None
        child_result = None
        while True:
            ret, frame = cap.read()

            # the resized version of the frame will be used for analysis
            resized = cv2.resize(frame, (width, height))

            if args.synchronous:
                child_result = []
                analyze_frame(resized, child_result, engine, labels)
            else:
                if not is_processing:
                    # kick off analysis in subprocess, if not currently analyzing
                    is_processing = True
                    next_child_result = []
                    child = multiprocessing.Process(target=analyze_frame, args=(resized, child_result, engine, labels))
                    child.start()
                elif child is not None and not child.is_alive():
                    child.join()
                    child_result = next_child_result

            frame = draw_boxes_on_frame(child_result, frame)

            if args.local:
                # weird black magic needed to draw to the screen
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.imshow('frame', frame)
            else:
                sys.stdout.buffer.write(frame.tobytes())


    except KeyboardInterrupt:
        print('Shutting down...')
    finally:
        cap.release()
        cv2.destroyAllWindows()

#Pass in the numpy array of the frame and the detection engine
#output is (boxes, frame) where
#boxes is a list of bounding boxes added
#frame is the stillframe with the boxes added
def analyze_frame(frame, boxes, engine, labels):
    img = Image.fromarray(frame)
    ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
        relative_coord=False, top_k=10)
    _, width, height, _ = engine.get_input_tensor_shape()
    for obj in ans:
        print ('-----------------------------------------')
        if labels:
            print(labels[obj.label_id])
        print ('score = ', obj.score)
        box = obj.bounding_box.flatten().tolist()
        print ('box = ', box)
        relative_box = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
        print ('relative_box = ', relative_box)
        # Draw a rectangle.
        if label_is_cat(obj.label_id):
            boxes.append(relative_box)
    # draw_boxes_on_picture(boxes, img)
    # out = np.array(img)

#draws the specified boxes in red on the input picture
def draw_boxes_on_picture(boxes, img):
    height = len(img)
    width = len(img[0])
    draw = ImageDraw.Draw(img)
    for box in boxes:
        absoluteBox = (int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) 
        draw.rectangle(absoluteBox, outline='red')

def draw_boxes_on_frame(boxes, frame):
    img = Image.fromarray(frame)
    draw_boxes_on_picture(boxes, img)
    return np.array(img)


def label_is_cat(label_id):
    return (label_id == 16) or (label_id == 17) or (label_id == 73) or (label_id == 87)

if __name__ == '__main__':
    main()
