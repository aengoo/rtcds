from data import cfg_re50, cfg_mnet
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.model_utils import *
from utils.timer import Timer
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
import torch.backends.cudnn as cudnn
import numpy as np
import os

# torch, cudnn settings
torch.set_grad_enabled(False)
cudnn.benchmark = True

class Detector:
    def __init__(self, weight_path: str, timer: Timer = None, model: str = 're50'):
        # configure backbone network
        """
        model : re50 or mnet
        """

        if model == 're50':
            self.cfg = cfg_re50
        elif model == 'mnet':
            self.cfg = cfg_mnet

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, os.path.join(weight_path), False)
        self.net.eval()
        print('Finished loading model!')
        # print(net)
        self.device = torch.device("cuda")
        self.net = self.net.to(self.device)
        self.timer = Timer() if timer is None else timer

    def run(self, img_tensor, vectorized=True, conf_thresh: float = 0.5, nms_thr: float = 0.4):
        self.timer.tic()
        img = np.float32(img_tensor)
        # im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)

        priorbox = PriorBox(self.cfg, image_size=tuple(img_tensor.shape[:2]))
        if vectorized:
            priors = priorbox.vectorized_forward()
        else:
            priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_thr)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        self.timer.toc()
        return dets

    def get_time_str(self):
        # returns computing time for last operation and average
        return f'{self.timer.diff:.4f}', f'{self.timer.average_time:.4f}'

