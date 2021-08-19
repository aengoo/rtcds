from __future__ import print_function
import argparse

from modules.detector import Detector
from utils.timer import Timer
from utils.general import *
from tracker.sort import *
import os
import copy
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='../save/test/raw', help='')
parser.add_argument('--vid-path', type=str, default='../data/test/last', help='')
parser.add_argument('--label-path', type=str, default='last_label', help='')
parser.add_argument('--save-path', type=str, default='last_vids', help='')
OPT = parser.parse_args()

vid_list = os.listdir(os.path.join(OPT.vid_path))

save_dir = os.path.join(OPT.data, OPT.save_path)
os.makedirs(save_dir, exist_ok=True)

for vid_idx, vid_name in enumerate(vid_list):
    with open(os.path.join(OPT.data, OPT.label_path, str(vid_name) + '.txt'), 'r') as f:
        lines = f.readlines()
        label_dict = {}
        for line in lines:
            tmp_line = line.replace('\n', '')
            label_dict.update({int(tmp_line.split(', ')[0]): tmp_line.split(', ')[1:]})

        save_path = os.path.join(save_dir, vid_name)
        vid_save = None
        vid_writer = None

        # stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)
        stream = cv2.VideoCapture(os.path.join(OPT.vid_path, vid_name))
        total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

        print(f'[{vid_idx + 1:d}/{len(vid_list):d}] Processing {vid_name:s}...')

        # for frame_idx, frame_data in enumerate(stream):
        now_frame = 0
        while now_frame <= total_frame:
            # img_name, img_rgb, img_raw, vid_cap = frame_data
            ret, img_raw = stream.read()
            if ret:
                if now_frame in label_dict.keys():
                    bbox = label_dict[now_frame]
                    box = [int(b) for b in bbox[1:]]
                    plot_one_box(box, img_raw, color=[255, 0, 0], label=str(bbox[0]))

                # cv2.imshow('frame', img_raw)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                if vid_save != save_path:
                    vid_save = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fps = stream.get(cv2.CAP_PROP_FPS)
                    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img_raw)
                now_frame += 1
            else:
                total_frame -= 1
        stream.release()

