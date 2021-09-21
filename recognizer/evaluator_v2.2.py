from __future__ import print_function
import argparse
from modules.detector_new import Detector
from modules.identifier_new import Identifier
from utils.data_utils import *
from utils.general import get_iou
from utils.eval_util import NewCounter
from utils.timer import Timer
from utils.label_reader import *
from tracker.sort import *
from modules.logger import Logger
import copy
import math
import pandas as pd


AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


def box_adapt(box: list, res: tuple, rat=1.):
    w, h = (box[2] - box[0], box[3] - box[1])
    if rat != 1.:
        pad = [- ((w * rat) - w) / 2, - ((h * rat) - h) / 2, ((w * rat) - w) / 2, ((h * rat) - h) / 2]
        new_box = [(xy + pad[idx]) * res[idx % 2] for idx, xy in enumerate(box[:4])] + box[4:]
    else:
        new_box = [xy * res[idx % 2] for idx, xy in enumerate(box[:4])] + box[4:]
    return new_box


def get_box_diagonal(xyxy):
    w, h = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return math.sqrt((w**2) + (h**2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data', help='')
    parser.add_argument('--t-faces', type=str, default='target/faces-17', help='')
    parser.add_argument('--k-faces', type=str, default='target/faces-400', help='')
    parser.add_argument('--n-faces', type=int, default=20, help='')
    parser.add_argument('--vid-path', type=str, default='test/vid_final', help='')
    parser.add_argument('--label-path', type=str, default='test/label_final', help='')
    parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
    parser.add_argument('--vid-res', type=str, default='FHD', help='among FHD, HD, sHD and VGA')
    parser.add_argument('--box-ratio', type=float, default=1.10, help='')
    parser.add_argument('--mono-res', action='store_true', help='')
    parser.add_argument('--name', type=str, default='result', help='')
    parser.add_argument('--many-landms', action='store_true', help='works as 68 landmarks')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='')
    parser.add_argument('--event-thresh', type=int, default=10, help='')
    parser.add_argument('--ts-thresh', type=float, default=3.5, help='')
    parser.add_argument('--repeat', type=int, default=1, help='1 means operate only once')

    OPT = parser.parse_args()

    # label_vid_check
    vid_list = os.listdir(str(os.path.join(OPT.data, OPT.vid_path)))
    label_list = os.listdir(str(os.path.join(OPT.data, OPT.label_path)))
    trimmed_label_list = [label_name.replace('.txt', '') for label_name in label_list]
    vid_list.sort()
    trimmed_label_list.sort()
    if vid_list != trimmed_label_list:
        print('label-vid check detected some problem!')
        exit()
    else:
        print('label-vid check completed with no problem!')

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

    frame_counter = NewCounter()
    event_counter = NewCounter()

    for i in range(OPT.repeat):
        for vid_idx, vid_name in enumerate(vid_list):
            print(f'[{vid_idx+1:d}/{len(vid_list):d}] Processing...')

            # stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)
            stream = cv2.VideoCapture(os.path.join(OPT.data, OPT.vid_path, vid_name))
            total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

            vid_gt_name = vid_name.split('_')[0]
            vid_label = VideoLabel(os.path.join(OPT.data, OPT.label_path, str(vid_name) + '.txt'))

            tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

            identifier.set_random_faces(vid_name.split('_')[0])

            frame_idx = 0

            ts_thres = OPT.ts_thresh
            track_dict = {}
            event_raised = False

            while frame_idx <= total_frame:
                overall_timer.tic()
                ret, img_raw = stream.read()
                if ret:
                    img_det = cv2.resize(img_raw, (640, 360)) if not OPT.mono_res else copy.deepcopy(img_raw)

                    boxes = detector.run(img_det)

                    # get ground-truth boxes of each frame
                    gt = vid_label.get_gt_boxes(frame_idx)[0]
                    if len(gt):
                        box_gt_name = gt[0]
                        gt_box = box_adapt(gt[1:], res=AVAILABLE_RESOLUTIONS[OPT.vid_res], rat=OPT.box_ratio)
                    else:
                        box_gt_name = '-'
                        gt_box = None

                    identified = identifier.run(img_raw, boxes)
                    # identified : [(box, score, idt, face_name, face_dist, face_std_score), ...]
                    for box, score, idt, face_name, face_dist, face_std_score in identified:
                        if gt_box:
                            if get_iou(gt_box, box) > OPT.iou_thresh:
                                frame_counter.count(box_gt_name, face_name)
                            else:
                                frame_counter.count('-', face_name)
                        else:
                            frame_counter.count('-', face_name)

                    tracked = tracker.update(boxes)
                    track_identified = identifier.run(img_raw, tracked)
                    for box, score, idt, face_name, face_dist, face_std_score in track_identified:
                        total_score = ((score + face_std_score + (1 - (face_dist*1.65))) * (1. / ts_thres)) ** 2
                        if idt in track_dict:
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
                                track_dict.update({idt: {'name': face_name, 'score': total_score, 'raised': False}})
                            else:
                                track_dict.update({idt: {'name': face_name, 'score': 0., 'raised': False}})

                    for idt in track_dict.keys():
                        if track_dict[idt]['score'] > OPT.event_thresh and not track_dict[idt]['raised']:
                            track_dict[idt]['raised'] = True
                            event_counter.count(vid_gt_name, track_dict[idt]['name'])
                            event_raised = True

                    frame_idx += 1
                else:
                    total_frame -= 1
            if not event_raised:
                event_counter.count(vid_gt_name, '-')
            stream.release()
    logger = Logger(save_path=os.path.join(OPT.data, 'results'))
    logger.log_codes()
    logger.print_new_eval(counter=frame_counter, is_save=False, print_name='frame')
    logger.print_new_eval(counter=event_counter, is_save=False, print_name='track')
    logger.print_new_eval(counter=frame_counter, is_save=True, print_name='frame')
    logger.print_new_eval(counter=event_counter, is_save=True, print_name='track')
    logger.print_args(OPT, is_save=True)

