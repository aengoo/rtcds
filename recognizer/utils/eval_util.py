import pandas as pd
import numpy as np
import copy


class Counter:
    def __init__(self, num_cls: int, cls_indices: list):
        self.num_cls = num_cls
        self.cls_indices = cls_indices
        self.conf_mat = pd.DataFrame()
        self.refresh()

    def refresh(self):
        np_mat = np.zeros((self.num_cls, self.num_cls + 1), dtype=int)
        self.conf_mat = pd.DataFrame(np_mat, self.cls_indices, self.cls_indices + ['-'])

    def count(self, ground_truth: str, predicted: str):
        self.conf_mat[predicted][ground_truth] += 1

    def save_csv(self, save_path: str):
        self.conf_mat.to_csv(path_or_buf=save_path, sep=',', na_rep='NaN')

    def get_mat(self):
        return copy.deepcopy(self.conf_mat)

    def get_accuracy(self):
        di = np.diag_indices(self.num_cls)
        return self.conf_mat.to_numpy()[di].sum() / self.conf_mat.to_numpy().sum()

    def get_single_precision(self, cls_name: str):
        return self.conf_mat[cls_name][cls_name] / self.conf_mat[cls_name].sum()

    def get_single_recall(self, cls_name: str):
        return self.conf_mat[cls_name][cls_name] / self.conf_mat.T[cls_name].sum()

    def get_multi_precision(self):
        # returns 'literally' averaged precision of all classes
        return sum(self.get_single_precision(cls_name=cls) for cls in self.cls_indices)/self.num_cls

    def get_multi_recall(self):
        # returns 'literally' averaged recall of all classes
        return sum(self.get_single_recall(cls_name=cls) for cls in self.cls_indices)/self.num_cls

    def get_f1_score(self):
        precision = self.get_multi_precision()
        recall = self.get_multi_recall()
        return (2 * precision * recall) / (precision + recall)


'''
    def get_AP(self):
        pass

'''