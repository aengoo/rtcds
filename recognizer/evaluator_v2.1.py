from __future__ import print_function
import argparse
from modules.detector_new import Detector
from modules.identifier_new import Identifier
from utils.data_utils import *
from utils.general import get_iou
from utils.timer import Timer
from utils.label_reader import *
from tracker.sort import *
from modules.logger import Logger
import copy
import math
import pandas as pd


def get_box_diagonal(xyxy):
    w, h = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return math.sqrt((w**2) + (h**2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data', help='')
    parser.add_argument('--faces', type=str, default='target/faces-17', help='')
    parser.add_argument('--vid-path', type=str, default='test/vid_final', help='')
    parser.add_argument('--label-path', type=str, default='test/label_final', help='')
    parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
    parser.add_argument('--vid-res', type=str, default='FHD', help='among FHD, HD, sHD and VGA')
    parser.add_argument('--box-ratio', type=float, default=1.2, help='')
    parser.add_argument('--mono-res', action='store_true', help='')
    parser.add_argument('--no-trk', action='store_true', help='do not apply tracking')
    parser.add_argument('--eval-name', type=str, default='result', help='')
    parser.add_argument('--many-landms', action='store_true', help='works as 68 landmarks')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='')

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

    identifier = Identifier(face_path=os.path.join(OPT.data, OPT.faces),
                            idt_res=OPT.vid_res,
                            box_ratio=OPT.box_ratio,
                            is_eval=True,
                            timer=idt_timer,
                            landms68=OPT.many_landms)

    dist_list = []

    for vid_idx, vid_name in enumerate(vid_list):
        print(f'[{vid_idx+1:d}/{len(vid_list):d}] Processing...')

        # stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)
        stream = cv2.VideoCapture(os.path.join(OPT.data, OPT.vid_path, vid_name))
        total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

        vid_label = VideoLabel(os.path.join(OPT.data, OPT.label_path, str(vid_name) + '.txt'))

        tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3) if not OPT.no_trk else None  # refreshed for each video

        frame_idx = 0
        while frame_idx <= total_frame:
            overall_timer.tic()
            ret, img_raw = stream.read()
            if ret:
                img_det = cv2.resize(img_raw, (640, 360)) if not OPT.mono_res else copy.deepcopy(img_raw)

                boxes = detector.run(img_det)

                # get ground-truth boxes of each frame
                gt = vid_label.get_gt_boxes(frame_idx)[0]
                if len(gt):
                    gt_name = gt[0]
                    gt_box = gt[1:]
                else:
                    gt_name = '-'
                    gt_box = None

                if tracker:
                    boxes = tracker.update(boxes)

                identified = identifier.run(img_raw, boxes)
                # identified : [(box, score, idt, face_name, face_dist, face_std_score), ...]
                for box, score, idt, face_name, face_dist, face_std_score in identified:
                    if gt_name == '-':
                        if face_name != '-':
                            dist_list.append([score, get_box_diagonal(box), face_dist, face_std_score, 'False_Match'])
                    else:
                        if get_iou(gt_box, box) >= OPT.iou_thresh:
                            if gt_name == face_name:
                                dist_list.append([score, get_box_diagonal(box), face_dist, face_std_score, 'True_Match'])
                            elif face_name != '-':
                                dist_list.append([score, get_box_diagonal(box), face_dist, face_std_score, 'MisMatch'])
                        else:
                            if face_name != '-':
                                dist_list.append([score, get_box_diagonal(box), face_dist, face_std_score, 'False_Match'])
                overall_timer.toc()
                frame_idx += 1
            else:
                total_frame -= 1
        stream.release()

    logger = Logger(save_path=os.path.join(OPT.data, 'results'))
    df_indices = ['det_conf', 'box_diagonal', 'face_dist', 'face_std_score', 'confusion']
    # confusion : True_Match, (False_UnMatch), False_Match, (True_UnMatch), MisMatch
    dist_df = pd.DataFrame(np.array(dist_list), columns=df_indices)
    logger.log_codes()
    logger.log_dataframe(dist_df, 'dist.csv', save_plt=True, hue_key='confusion')
    logger.print_args(OPT, is_save=True)

