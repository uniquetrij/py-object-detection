from threading import Thread
import cv2
from py_flask_movie.flask_movie import FlaskMovie
from py_object_detection.tf_api.tf_object_detection_api import TFObjectDetector, \
    PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, LABELMAP_mscoco
from py_pipe.pipe import Pipe
from py_tensorflow_runner.session_utils import SessionRunner


def detect_objects(cap, pipe, detector, default):
    if not default:
        ret_pipe = Pipe()
    else:
        ret_pipe = None

    def start_cam():
        while True:
            ret, image = cap.read()
            if not ret:
                continue
            inference = TFObjectDetector.Inference(image.copy(), LABELMAP_mscoco, return_pipe=ret_pipe)
            detector.get_in_pipe().push_wait()
            detector.get_in_pipe().push(inference)

    Thread(target=start_cam).start()
    while True:
        if not default:
            ret, inference = ret_pipe.pull(True)
            if not ret:
                ret_pipe.pull_wait()
            else:
                ret_pipe.flush()
        else:
            detector.getOutPipe().pull_wait()
            ret, inference = detector.getOutPipe().pull(True)
        if ret:
            i_dets = inference.get_result()
            pipe.push(i_dets.get_annotated())


if __name__ == '__main__':

    fs = FlaskMovie()
    fs.start("0.0.0.0", 5000)

    session_runner = {}
    detector = {}

    cap = {}
    pipe = {}
    video_inputs = {0: 0, 1: 1}

    for i in video_inputs.keys():
        session_runner[i] = SessionRunner()
        session_runner[i].start()
        detector[i] = TFObjectDetector(PRETRAINED_ssd_mobilenet_v1_coco_2017_11_17, LABELMAP_mscoco, None,
                                           'tf_api_' + str(i), True)
        detector[i].use_session_runner(session_runner[i])
        detector[i].run()

        cap[i] = cv2.VideoCapture(video_inputs[i])
        pipe[i] = Pipe()
        fs.create('feed_' + str(i), pipe[i])
        Thread(target=detect_objects, args=(cap[i], pipe[i], detector[i], False)).start()
