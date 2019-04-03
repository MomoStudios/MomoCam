python3 classify_capture_opencv.py --label ~/Downloads/canned_models/coco_labels.txt --model ~/Downloads/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --synchronous | ffmpeg -f rawvideo -r 60 -s 640x480 -pix_fmt bgr24 -i - -c:v libx264 -preset fast -pix_fmt yuv420p -s 640x480 -g 30 -f flv rtmp://localhost/live/cat