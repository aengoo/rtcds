import pandas as pd
import numpy as np


class Counter:
    def __init__(self, num_cls: int, cls_indices: list):
        self.num_cls = num_cls
        self.cls_indices = cls_indices
        self.conf_mat = None
        self.refresh()

    def refresh(self):
        self.conf_mat = pd.DataFrame(np.zeros((self.num_cls, self.num_cls + 1), dtype=int), self.cls_indices,
                                     self.cls_indices + ['-'])

    def count(self, ground_truth: str, predicted: str):
        self.conf_mat[predicted][ground_truth] += 1

    def save_csv(self):
        pass

    def print_mat(self):
        print(self.conf_mat)
