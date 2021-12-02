from __future__ import print_function
import argparse
from modules.detector import Detector
from utils.general import *
import os
import copy
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='test/raw', help='input video path')
parser.add_argument('--weight-path', type=str, default='weights/Resnet50_Final.pth', help='pth file path')
parser.add_argument('--save-path', type=str, default='../output', help='output video path')
parser.add_argument('--save-video', action='store_true', help='select if you want to save videos')
OPT = parser.parse_args()

detector = Detector(weight_path=OPT.weight_path, model='re50')

vid_list = os.listdir(OPT.video_path)
save_dir = os.path.join(OPT.save_path, OPT.video_path)
os.makedirs(save_dir, exist_ok=True)

for vid_idx, vid_name in enumerate(vid_list):
    save_path = os.path.join(save_dir, vid_name)
    vid_save = None
    vid_writer = None

    stream = cv2.VideoCapture(os.path.join(OPT.video_path, vid_name))
    total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f'[{vid_idx + 1:d}/{len(vid_list):d}] Processing {vid_name:s}...', end='')

    now_frame = 0
    while now_frame <= total_frame:
        ret, img_raw = stream.read()
        if ret:
            img_tensor = copy.deepcopy(img_raw)

            dets = detector.run(img_tensor)
            for bbox in dets:
                box = [int(b) for b in bbox[:4]]
                plot_one_box(box, img_raw, color=[255, 0, 0])
                with open(str(save_path) + '.csv', 'a') as f:
                    f.write(f'{now_frame:d},{box[0]:d},{box[1]:d},{box[2]:d},{box[3]:d}\n')

            if OPT.save_video:
                if vid_save != save_path:
                    vid_save = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # releasing previous video writer
                    fps = stream.get(cv2.CAP_PROP_FPS)
                    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img_raw)
            now_frame += 1
    stream.release()


