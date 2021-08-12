from __future__ import print_function
import argparse
from modules.detector import Detector
from modules.identifier import Identifier
from utils.data_utils import *
from utils.timer import Timer
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='')
parser.add_argument('--faces', type=str, default='target/faces-4', help='')
parser.add_argument('--vid-path', type=str, default='test/temp', help='')
parser.add_argument('--weight', type=str, default='weights/Resnet50_Final.pth', help='')
parser.add_argument('--vid-res', type=str, default='FHD', help='among FHD, HD, sHD and VGA')
parser.add_argument('--bbox-pad', type=int, default=10, help='')
parser.add_argument('--tsr', type=int, default=10, help='')
parser.add_argument('--dual-res', action='store_true', help='')
parser.add_argument('--no-trk', action='store_true', help='do not apply tracking')
parser.add_argument('--eval-name', type=str, default='result', help='')
parser.add_argument('--trk-timing', type=str, default='default', help='default, endpoint, number(digit)')
parser.add_argument('--many-lm', action='store_true', help='works as 68 landmarks')
parser.add_argument('--conf_thresh', type=float, default=0.995, help='')
# TODO: conf_thres test

OPT = parser.parse_args()
if OPT.dual_res and (OPT.vid_res == 'VGA'):
    exit()

overall_timer, det_timer, idt_timer = (Timer(), Timer(), Timer())

detector = Detector(weight_path=os.path.join(OPT.data, OPT.weight), model='re50', timer=det_timer)
identifier = Identifier(face_path=os.path.join(OPT.data, OPT.faces),
                        det_res='sHD' if OPT.dual_res else OPT.vid_res,
                        idt_res=OPT.vid_res,
                        bbox_pad=OPT.bbox_pad,
                        tsr=OPT.tsr,
                        evaluation=True,
                        tracking=not OPT.no_trk,
                        timer=idt_timer,
                        trk_timing=OPT.trk_timing,
                        mlm=OPT.many_lm)

vid_list = os.listdir(os.path.join(OPT.data, OPT.vid_path))
for vid_idx, vid_name in enumerate(vid_list):
    print(f'[{vid_idx+1:d}/{len(vid_list):d}] Processing...')
    stream = LoadImages(os.path.join(OPT.data, OPT.vid_path, vid_name), print_info=False)

    for frame_idx, frame_data in enumerate(stream):
        overall_timer.tic()
        img_name, img_rgb, img_raw, vid_cap = frame_data
        img_tensor = cv2.resize(img_raw, (640, 360)) if OPT.dual_res else copy.deepcopy(img_raw)

        dets = detector.run(img_tensor)
        identifier.run(img_raw, dets, gt_name=str(vid_name).split('_')[0])
        overall_timer.toc()

        # cv2.imshow('test', idts)
        # if cv2.waitKey(1) == ord('q'):
        #     raise StopIteration
    if OPT.trk_timing == 'endpoint':
        identifier.count_endpoint(gt_name=str(vid_name).split('_')[0])

print('options:', OPT)
counter = identifier.get_evaluator()
print(counter.get_mat())
print('Acc:', counter.get_accuracy())
print('Each-Precision:', [cls + format(counter.get_single_precision(cls), '.4f') for cls in counter.cls_indices])
print('Multi-Precision:', counter.get_multi_precision())
print('Multi-Recall:', counter.get_multi_recall())
print('F1-score:', counter.get_f1_score())

# print('PR:', counter.get_PR())
counter.save_csv(save_path=os.path.join(OPT.data, 'results', 'mat', OPT.eval_name + '.csv'))
print(f'det_time: {det_timer.average_time:.3f}')
print(f'idt_time: {idt_timer.average_time:.3f}')
print(f'overall_time: {overall_timer.average_time:.3f}')
with open(os.path.join(OPT.data, 'results', 'txt', OPT.eval_name + '.txt'), 'a') as f:
    print('options:', OPT, file=f)
    print(counter.get_mat(), file=f)
    print('Acc:', counter.get_accuracy(), file=f)
    print('Each-Precision:', [cls + format(counter.get_single_precision(cls), '.4f') for cls in counter.cls_indices], file=f)
    print('Multi-Precision:', counter.get_multi_precision(), file=f)
    print('Multi-Recall:', counter.get_multi_recall(), file=f)
    print('F1-score:', counter.get_f1_score(), file=f)
    f.write(f'det_time: {det_timer.average_time:.3f}\n')
    f.write(f'idt_time: {idt_timer.average_time:.3f}\n')
    f.write(f'overall_time: {overall_timer.average_time:.3f}\n')


