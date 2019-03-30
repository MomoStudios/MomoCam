"""
Usage:

python3 demo/classify_capture_opencv.py \
    --model test_data/inception_v4_299_quant_edgetpu.tflite  \
    --label test_data/imagenet_labels.txt \
    [--synchronous]

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

            frame = cv2.resize(frame, (width, height))

            if args.synchronous:
                child_result = []
                analyzed_frame(frame, child_result, engine)
            else:
                if not is_processing:
                    # kick off analysis in subprocess, if not currently analyzing
                    is_processing = True
                    child_result = []
                    child = multiprocessing.Process(target=analyze_frame, args=(frame, child_result))
                    child.start()
                elif child is not None and not child.is_alive():
                    child.join()
                    # child has finished processing, grab results
                    # TODO:figure out what to do with the results, they'll need to be added to the raw frame we send to stdout
                    pass

            sys.stdout.buffer.write(frame.tostring())

    except Exception as e:
        print('thats no good!')
        print(e)
    except KeyboardInterrupt:
        print('Shutting down...')
    finally:
        cap.release()

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
