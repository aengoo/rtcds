from __future__ import print_function
import argparse
from modules.detector_new import Detector
from modules.identifier_new import Identifier
from utils.data_utils import *
from utils.general import *
from utils.eval_util import NewCounter
from utils.timer import Timer
from utils.label_reader import *
from tracker.sort import *
from modules.logger import Logger
import copy
import math
import pandas as pd

import base64
import json
import redis


AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


def box_adapt(box: list, rat=1.):
    w, h = (box[2] - box[0], box[3] - box[1])
    if rat != 1.:
        pad = [- ((w * rat) - w) / 2, - ((h * rat) - h) / 2, ((w * rat) - w) / 2, ((h * rat) - h) / 2]
        new_box = [(xy + pad[idx]) for idx, xy in enumerate(box[:4])] + box[4:]
    else:
        new_box = [xy for idx, xy in enumerate(box[:4])] + box[4:]
    return new_box


def get_box_diagonal(xyxy):
    w, h = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return math.sqrt((w**2) + (h**2))


def lpush_frame(img, rdo: redis.StrictRedis):
    img = json.dumps(img).encode('utf-8')  #Object of type 'bytes' is not JSON serializable
    rdo.lpush("vid", img)
    rdo.ltrim("vid", 0, 29)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data', help='')
    parser.add_argument('--t-faces', type=str, default='target/faces-17', help='')
    parser.add_argument('--k-faces', type=str, default='target/faces-400', help='')
    parser.add_argument('--n-faces', type=int, default=20, help='')
    parser.add_argument('--vid-path', type=str, default='test/vid_celeb', help='')
    parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
    parser.add_argument('--vid-res', type=str, default='VGA', help='among FHD, HD, sHD and VGA')
    parser.add_argument('--box-ratio', type=float, default=1.30, help='')
    parser.add_argument('--mono-res', action='store_true', help='')
    parser.add_argument('--save-vid', action='store_true', help='')
    parser.add_argument('--name', type=str, default='result', help='')
    parser.add_argument('--many-landms', action='store_true', help='works as 68 landmarks')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='')
    parser.add_argument('--event-thresh', type=int, default=10, help='')
    parser.add_argument('--ts-thresh', type=float, default=4.0, help='')
    parser.add_argument('--repeat', type=int, default=1, help='1 means operate only once')

    OPT = parser.parse_args()

    # label_vid_check
    vid_list = os.listdir(str(os.path.join(OPT.data, OPT.vid_path)))

    overall_timer, det_timer, idt_timer = (Timer(), Timer(), Timer())

    detector = Detector(weight_path=os.path.join(OPT.data, OPT.weight),
                        model='re50',
                        timer=det_timer,
                        conf_thresh=OPT.conf_thresh)

    identifier = Identifier(target_path=os.path.join(OPT.data, OPT.t_faces),
                            other_path=os.path.join(OPT.data, OPT.k_faces),
                            n=OPT.n_faces,
                            idt_res=OPT.vid_res,
                            box_ratio=OPT.box_ratio,
                            is_eval=True,
                            timer=idt_timer,
                            landms68=OPT.many_landms)

    logger = Logger(save_path=os.path.join(OPT.data, 'results'))
    cap = cv2.VideoCapture(0)
    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.5)

    identifier.set_all_random_faces()

    frame_idx = 0

    ts_thres = OPT.ts_thresh
    track_dict = {}
    event_raised = False

    rd = redis.StrictRedis(host='localhost', port=6379, db=0)
    rd.flushdb()  # db 초기화

    while True:
        overall_timer.tic()
        ret, img_raw = cap.read()
        if ret:
            img_det = cv2.resize(img_raw, (640, 360)) if not OPT.mono_res else copy.deepcopy(img_raw)

            boxes = np.clip(detector.run(img_det), 0., 1.)

            identified = identifier.run(img_raw, boxes)
            # identified : [(box, score, idt, face_name, face_dist, face_std_score), ...]

            tracked = tracker.update(boxes)
            track_identified = identifier.run(img_raw, tracked)
            for box, score, idt, face_name, face_dist, face_std_score in track_identified:
                total_score = ((score + face_std_score + (1 - (face_dist*1.65))) * (1. / ts_thres)) ** 3
                if idt in track_dict:
                    track_dict[idt]['last_box'] = box
                    track_dict[idt]['plotted'] = False
                    if face_name != '-':
                        if track_dict[idt]['name'] == face_name:
                            track_dict[idt]['score'] += total_score
                        elif track_dict[idt]['score'] < total_score:
                            track_dict[idt]['name'] = face_name
                            track_dict[idt]['score'] = abs(track_dict[idt]['score'] - total_score)
                            track_dict[idt]['raised'] = False
                        else:
                            track_dict[idt]['score'] -= total_score
                else:
                    if face_name != '-':
                        track_dict.update({idt: {'name': face_name, 'score': total_score, 'raised': False, 'last_box': box, 'plotted': False}})
                    else:
                        track_dict.update({idt: {'name': face_name, 'score': 0., 'raised': False, 'last_box': box, 'plotted': False}})

            for idt in track_dict.keys():
                if track_dict[idt]['score'] > OPT.event_thresh and not track_dict[idt]['raised']:
                    track_dict[idt]['raised'] = True
                    event_raised = True

            if OPT.save_vid:
                for box, score, idt, face_name, face_dist, face_std_score in identified:
                    plot_tp_box(box, img_raw, 0.5, 'g')
                for idt in track_dict.keys():
                    if not track_dict[idt]['plotted']:
                        if track_dict[idt]['raised']:
                            plot_one_box(track_dict[idt]['last_box'], img_raw, label=track_dict[idt]['name'])
                        else:
                            plot_one_box(track_dict[idt]['last_box'], img_raw)
                        track_dict[idt]['plotted'] = True

                retval, buffer = cv2.imencode('.jpg', img_raw)
                fr = base64.b64encode(buffer)  # Json dump를 위한 encoding #base64 encode read data, result = bytes
                fr = fr.decode('utf-8')
                lpush_frame(fr, rd)
                # cv2.imshow('test', img_raw)
                # if cv2.waitKey(1) == ord('q'):
                #    raise StopIteration
