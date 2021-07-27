import embed.embedding as face_encoder
import os
from tracker.sort import *
from utils.timer import Timer
from utils.general import *
import operator  # 나중에 없애야함 너무 비효율적
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary


AVAILABLE_RESOLUTIONS = {
    'FHD': (1080, 1920, 3),
    'HD': (720, 1280, 2),
    'sHD': (360, 640, 1),
    'VGA': (480, 640, 1),
}

class Identifier:
    def __init__(self, face_path: str, det_res: str, idt_res: str, bbox_pad: int, timer: Timer = None):
        self.encoder = face_encoder.EncodeFace(face_path)

        self.faces = []
        self.tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

        if idt_res == 'VGA':
            self.rat = 1
        else:
            self.rat = int(AVAILABLE_RESOLUTIONS[idt_res][2] / AVAILABLE_RESOLUTIONS[det_res][2])
        self.pad = bbox_pad * AVAILABLE_RESOLUTIONS[idt_res][2]
        self.timer = Timer() if timer is None else timer
        self.identified = {}

    def run(self, img_raw, boxes):
        self.timer.tic()
        face_boxes = []

        tracked = self.tracker.update(boxes)
        for tbox in tracked:
            id = int(tbox[4])
            box = [int(b) * self.rat for b in tbox[:4]]
            crop_box = [int(box[0]) - self.pad, int(box[1]) - self.pad, int(box[2]) + self.pad, int(box[3]) + self.pad]
            cropped = img_raw[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

            face_name = self.encoder.match_face(cropped)
            if id in self.identified:
                self.identified[id][face_name] += 1 if face_name == '-' else 10
            else:
                self.identified.update({id: {k: 0 for k in self.encoder.get_faces_name() + ['-']}})
                self.identified[id][face_name] += 1 if face_name == '-' else 10
            face_boxes.append((box[:4], max(self.identified[id].items(), key=operator.itemgetter(1))[0]))

        [plot_one_box(box[0], img_raw, label=box[1]) for box in face_boxes]

        self.timer.toc()
        return img_raw

    def get_time_str(self):
        # returns computing time for last operation and average
        return f'{self.timer.diff:.4f}', f'{self.timer.average_time:.4f}'
