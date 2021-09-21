import numpy as np

import modules.encoder as face_encoder
from utils.timer import Timer
from utils.eval_util import *
from utils.general import *

AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


def box_adapt(boxes: np.ndarray, res: tuple, rat=1.):
    res_boxes = boxes[:, :4] * np.array(res[:2][::-1] + res[:2][::-1])
    boxes_shape = res_boxes[:, 2:] - res_boxes[:, :2]
    boxes_center = (boxes_shape / 2) + res_boxes[:, :2]
    pad_boxes_rad = (boxes_shape * rat if rat != 1. else boxes_shape) / 2.
    pad_res_boxes = np.concatenate((boxes_center - pad_boxes_rad, boxes_center + pad_boxes_rad, boxes[:, 4:]), axis=1, dtype='float32')
    return pad_res_boxes


class Identifier:
    def __init__(self, target_path: str, other_path: str, n: int, idt_res: str, box_ratio: float, timer: Timer = None,
                 is_eval=False, landms68: bool = False):

        self.encoder = face_encoder.EncodeFace(target_path=target_path, other_path=other_path, n=n, landmark68=landms68)
        self.faces = []
        self.res = AVAILABLE_RESOLUTIONS[idt_res]

        self.box_ratio = box_ratio
        self.timer = timer
        self.is_eval = is_eval

    def run(self, img_idt, boxes):

        if self.timer:
            self.timer.tic()

        idt_boxes = []
        adapted_boxes = box_adapt(boxes, self.res, self.box_ratio)
        for tbox in adapted_boxes:
            # 좌표형식 xyxy
            idt = -1
            if len(tbox) > 5:
                idt = int(tbox[5])
            box = [int(b) for b in tbox[:4]]
            score = tbox[4]
            cropped = img_idt[box[1]:box[3], box[0]:box[2]]
            face_name, face_dist, face_std_score = self.encoder.match_face(cropped, get_score=True)
            idt_boxes.append((box, score, idt, face_name, face_dist, face_std_score))

        if self.timer:
            self.timer.toc()

        if self.is_eval:
            return idt_boxes

        else:
            # tracking not applied
            [plot_center_text(box[0], img_idt, label=format(box[1], '.4f')) for box in idt_boxes]
            [plot_one_box(box[0], img_idt, label=box[3]) for box in idt_boxes]
            return img_idt

    def set_random_faces(self, gt_name):
        self.encoder.set_random_encodings(gt_name)
