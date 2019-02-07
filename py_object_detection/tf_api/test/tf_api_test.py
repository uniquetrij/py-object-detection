from threading import Thread
import cv2

from py_object_detection.tf_api.tf_object_detection_api import TFObjectDetector, \
    PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, LABELMAP_mscoco

from py_tensorflow_runner.session_utils import SessionRunner

cap = cv2.VideoCapture(-1)

session_runner = SessionRunner()
while True:
    ret, image = cap.read()
    if ret:
        break

detection = TFObjectDetector(PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, LABELMAP_mscoco, image.shape, 'tf_api',
                             True)
detector_ip = detection.get_in_pipe()
detector_op = detection.get_out_pipe()
detection.use_session_runner(session_runner)
session_runner.start()
detection.run()


def read():
    while True:
        detector_ip.push_wait()
        ret, image = cap.read()
        if not ret:
            continue
        inference = TFObjectDetector.Inference(image, LABELMAP_mscoco)
        detector_ip.push(inference)


def run():
    while True:
        detector_op.pull_wait()
        ret, inference = detector_op.pull(True)
        if ret:
            i_dets = inference.get_result()
            cv2.imshow("annotated", i_dets.get_annotated())
            cv2.waitKey(1)


Thread(target=run).start()
Thread(target=read).start()
