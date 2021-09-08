import modules.encoder as face_encoder
import os
from tracker.sort import *
from utils.timer import Timer
from utils.general import *
from utils.eval_util import *
import operator  # 나중에 없애야함 너무 비효율적
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary


AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}


class Identifier:
    def __init__(self, face_path: str, det_res: str, idt_res: str, bbox_pad: int, tsr: int, timer: Timer = None,
                 evaluation=False, tracking: bool = True, trk_timing='default', mlm: bool = False,
                 conf_thresh: float = 0.0, eval_mode: str = '', iou_thresh: float = 0.3):
        """
        face_path : path of face images for face matching(identification)
        det_res : detection resolution
        idt_res : identification resolution
        bbox_pad : padding size for bounding boxes
        tsr : Tracking Score Ratio
        evaluation : evaluation mode activation
        tracking :tracking activation
        trk_timing : counting timing works with tracking
        mlm : many(68) landmarks
        """
        self.encoder = face_encoder.EncodeFace(face_path, mlm)

        self.faces = []
        self.tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

        if idt_res == 'VGA':
            self.rat = 1
        else:
            self.rat = int(AVAILABLE_RESOLUTIONS[idt_res][2] / AVAILABLE_RESOLUTIONS[det_res][2])
        self.pad = bbox_pad * AVAILABLE_RESOLUTIONS[idt_res][2]
        self.timer = Timer() if timer is None else timer
        self.identified = {}
        self.tsr = tsr

        self.evaluation = evaluation
        if self.evaluation:
            self.evaluator = Counter(self.encoder.get_faces_cnt(), self.encoder.get_faces_names())
        self.tracking = tracking
        self.trk_timing = trk_timing
        self.conf_thresh = conf_thresh
        self.eval_mode = eval_mode
        self.iou_thresh = iou_thresh

    def run(self, img_raw, boxes, gt_name: str = None, gt_box=None):
        if self.evaluation and not (gt_name and self.eval_mode):
            exit('[ERROR] If evaluation mode is activated, Ground_Truth name is required to method: Identifier.run()')
        """
        gt_name : temp param for present evaluator TODO: bbox based evaluation...
        """
        self.timer.tic()
        face_boxes = []
        # boxes = boxes[boxes[:, 4] >= self.conf_thresh]
        tracked = self.tracker.update(boxes)
        for tbox in tracked:
            # 좌표형식 (xy,xy)
            id = int(tbox[5])
            box = [int(b * self.rat) for b in tbox[:4]]
            score = tbox[4]
            crop_box = [int(box[0]) - self.pad, int(box[1]) - self.pad, int(box[2]) + self.pad, int(box[3]) + self.pad]
            cropped = img_raw[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

            face_name, face_dist = self.encoder.match_face(cropped)
            face_name = '-' if score < self.conf_thresh else face_name
            if self.tracking:
                if id in self.identified:
                    self.identified[id][face_name] += 1 if face_name == '-' else self.tsr
                else:
                    self.identified.update({id: {k: 0 for k in self.encoder.get_faces_names() + ['-']}})
                    self.identified[id][face_name] += 1 if face_name == '-' else self.tsr
                face_boxes.append((crop_box[:4], max(self.identified[id].items(), key=operator.itemgetter(1))[0], score, face_dist))
            else:
                face_boxes.append((crop_box[:4], face_name, score, face_dist))

        self.timer.toc()
        if self.evaluation:
            if self.trk_timing == 'default':
                if self.eval_mode == 'existence':
                    [self.evaluator.count(gt_name, box[1]) for box in face_boxes]
                elif self.eval_mode == 'detection':
                    for box in face_boxes:
                        if not gt_box:
                            self.evaluator.count('-', box[1])
                        elif get_iou(box[0], gt_box) >= self.iou_thresh:
                            self.evaluator.count(gt_name, box[1])
                        else:
                            self.evaluator.count('-', box[1])
                elif self.eval_mode == 'check':
                    # [plot_center_text(box[0], img_raw, label=format(box[3], '.4f') if box[3] else '') for box in face_boxes]
                    [plot_center_text(box[0], img_raw, label=format(box[2], '.4f')) for box in face_boxes]
                    [plot_one_box(box[0], img_raw, label=box[1]) for box in face_boxes]
                    return img_raw
                elif self.eval_mode == 'scoring':
                    pass


        else:
            [plot_center_text(box[:4], img_raw, label=format(box[4], '.4f')) for box in boxes]
            [plot_one_box(box[0], img_raw, label=box[1]) for box in face_boxes]
            return img_raw

    def get_time_str(self):
        # returns computing time for last operation and average
        return f'{self.timer.diff:.4f}', f'{self.timer.average_time:.4f}'

    def get_evaluator(self):
        if self.evaluation:
            return self.evaluator

    def count_endpoint(self, gt_name):
        assert self.trk_timing == 'endpoint' and self.eval_mode == 'existence'

        if self.trk_timing == 'endpoint':
            for idt in self.identified.keys():
                self.evaluator.count(gt_name, max(self.identified[idt].items(), key=operator.itemgetter(1))[0])
        self.identified = {}
