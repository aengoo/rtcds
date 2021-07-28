from __future__ import print_function
import argparse
from detector import Detector
from identifier import Identifier
import os
import cv2
import copy
from utils.timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='')
parser.add_argument('--faces', type=str, default='target/faces-4', help='')
parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
parser.add_argument('--vid-res', type=str, default='VGA', help='among FHD, HD, sHD and VGA')
parser.add_argument('--bbox-pad', type=int, default=10, help='')
parser.add_argument('--dual-res', action='store_true', help='')
OPT = parser.parse_args()

if OPT.dual_res and (OPT.vid_res == 'VGA'):
    exit()

detector = Detector(weight_path=os.path.join(OPT.data, OPT.weight), model='re50')
identifier = Identifier(face_path=os.path.join(OPT.data, OPT.faces),
                        det_res='sHD' if OPT.dual_res else OPT.vid_res,
                        idt_res=OPT.vid_res,
                        bbox_pad=OPT.bbox_pad,
                        tsr=1)

cap = cv2.VideoCapture(0)
timer = Timer()

while True:
    timer.tic()
    ret, img_raw = cap.read()
    img_tensor = cv2.resize(img_raw, (640, 360)) if OPT.dual_res else copy.deepcopy(img_raw)
    dets = detector.run(img_tensor)
    idts = identifier.run(img_raw, dets)
    cv2.imshow('test', idts)
    if cv2.waitKey(1) == ord('q'):
        raise StopIteration
    timer.toc()
    print(f'{timer.diff:.4f}, {timer.average_time:.4f}/n')

    with open('time_result.txt', 'a') as f:
        tstr = format(timer.diff, '.4f') + ' ' + detector.get_time_str()[0] + ' ' + identifier.get_time_str()[0] + '\n'
        f.write(tstr)

