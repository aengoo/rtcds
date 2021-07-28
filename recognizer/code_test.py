import pandas as pd
import numpy as np

cls_indices = ['KIM', 'KANG', 'LIM', 'CHOI']
conf_mat = pd.DataFrame(np.zeros((4, 5), dtype=int), cls_indices, cls_indices+['-'])
print(conf_mat)

from utils.eval_util import *

cnt = Counter(4, cls_indices)
cnt.count('KIM', 'KANG')
cnt.count('KIM', 'KIM')
cnt.count('KANG', 'KANG')
cnt.count('CHOI', 'CHOI')
cnt.count('KIM', 'CHOI')


cnt.print_mat()
print('acc:', cnt.get_accuracy())
print('PR:', cnt.get_PR('KIM'))