from __future__ import print_function
import argparse
from utils.general import *
import os
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='sample', help='')
parser.add_argument('--label-path', type=str, default='output/sample', help='')
parser.add_argument('--save-path', type=str, default='output', help='')
OPT = parser.parse_args()

vid_list = os.listdir(os.path.join(OPT.video_path))

save_dir = OPT.save_path
os.makedirs(save_dir, exist_ok=True)

for vid_idx, vid_name in enumerate(vid_list):
    with open(os.path.join(OPT.label_path, str(vid_name) + '.txt'), 'r') as f:
        lines = f.readlines()
        label_dict = {}
        for line in lines:
            tmp_line = line.replace('\n', '')
            label_dict.update({int(tmp_line.split(', ')[0]): tmp_line.split(', ')[1:]})

        save_path = os.path.join(save_dir, vid_name)
        vid_save = None
        vid_writer = None

        stream = cv2.VideoCapture(os.path.join(OPT.video_path, vid_name))
        total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

        print(f'[{vid_idx + 1:d}/{len(vid_list):d}] Processing {vid_name:s}...')

        now_frame = 0
        while now_frame <= total_frame:
            ret, img_raw = stream.read()
            if ret:
                if now_frame in label_dict.keys():
                    bbox = label_dict[now_frame]
                    box = [int(b) for b in bbox[1:5]]
                    plot_one_box(box, img_raw, color=[255, 0, 0], label=str(bbox[5]))

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

