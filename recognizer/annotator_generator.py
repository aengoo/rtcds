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

parser.add_argument('--data', type=str, default='../data', help='')
parser.add_argument('--vid-path', type=str, default='test/raw', help='')
parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
parser.add_argument('--save', type=str, default='../save', help='')
OPT = parser.parse_args()

detector = Detector(weight_path=os.path.join(OPT.data, OPT.weight), model='re50')

timer = Timer()

two = []
zero = []

vid_list = os.listdir(os.path.join(OPT.data, OPT.vid_path))

save_dir = os.path.join(OPT.save, OPT.vid_path)
os.makedirs(save_dir, exist_ok=True)

for vid_idx, vid_name in enumerate(vid_list):
    save_path = os.path.join(save_dir, vid_name)
    vid_save = None
    vid_writer = None

    tracker = Sort(max_age=3, min_hits=0, iou_threshold=0.3)

    # stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)
    stream = cv2.VideoCapture(os.path.join(OPT.data, OPT.vid_path, vid_name))
    total_frame = stream.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f'[{vid_idx + 1:d}/{len(vid_list):d}] Processing {vid_name:s}...', end='')
    zero_flag = True
    two_flag = False

    # for frame_idx, frame_data in enumerate(stream):
    now_frame = 0
    while now_frame <= total_frame:
        # img_name, img_rgb, img_raw, vid_cap = frame_data
        ret, img_raw = stream.read()
        if ret:
            timer.tic()
            img_tensor = copy.deepcopy(img_raw)

            dets = detector.run(img_tensor)
            if dets.shape[0] > 0:
                zero_flag = False
                if dets.shape[0] > 1:
                    two_flag = True

            tracked = tracker.update(dets[:, :5])
            for bbox in tracked:
                box = [int(b) for b in bbox[:4]]
                plot_one_box(box, img_raw, color=[255, 0, 0], label=str(bbox[4]))
                with open(str(save_path) + '.txt', 'a') as f:
                    f.write(f'{now_frame:d}, {str(bbox[4]):s}, {box[0]:d}, {box[1]:d}, {box[2]:d}, {box[3]:d}\n')

            timer.toc()

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
            timer.toc()
            now_frame += 1
    stream.release()

    if zero_flag:
        zero.append(vid_name)
    if two_flag:
        two.append(vid_name)
    # print(stream.nframes, end=' ')
    print(f'{timer.average_time:.4f}')
    timer.clear()

print(zero)
print(two)
