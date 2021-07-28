from __future__ import print_function
import argparse
from modules.detector import Detector
from modules.identifier import Identifier
from utils.data_utils import *
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='')
parser.add_argument('--faces', type=str, default='target/faces-4', help='')
parser.add_argument('--vid-path', type=str, default='test/temp', help='')
parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
parser.add_argument('--vid-res', type=str, default='FHD', help='among FHD, HD, sHD and VGA')
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

vid_list = os.listdir(os.path.join(OPT.data, OPT.vid_path))
for vid_idx, vid_name in enumerate(vid_list):
    stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)

    for frame_idx, frame_data in enumerate(stream):
        img_name, img_rgb, img_raw, vid_cap = frame_data
        img_tensor = cv2.resize(img_raw, (640, 360)) if OPT.dual_res else copy.deepcopy(img_raw)

        dets = detector.run(img_tensor)
        idts = identifier.run(img_raw, dets)

        cv2.imshow('test', idts)
        if cv2.waitKey(1) == ord('q'):
            raise StopIteration

