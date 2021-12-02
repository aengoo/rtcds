from sklearn import datasets
import pandas as pd
import numpy as np
# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy
import os


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


data_path = '../prediction'
data_list = []
[data_list.append(data) if data.split('.')[-2].endswith('_det_embeds') else None for data in os.listdir(data_path)]

file_idx_list = []
embed_list = []
label_list = []
conf_list = []
dist_list = []
id_list = []
zs_list = []

for data_file in data_list:
    with open(os.path.join(data_path, data_file), 'r') as f:
        for line in f.readlines():
            file_idx_list.append([data_file.replace('embeds', 'idt'), int(line.split(',')[0])])
            embed_list.append([float(i) for i in line.split(',')[6:-1]])
            conf_list.append(float(line.split(',')[5]))
            label_list.append(data_file.split('_')[0])

old_file = ''
temp_file = None
for now_file, frame_idx in file_idx_list:
    if old_file != now_file:
        temp_file = [line.split(',') for line in open(os.path.join(data_path, now_file), 'r').readlines()]
        old_file = now_file
    for tks in temp_file:
        if int(tks[0]) == frame_idx:
            dist_list.append(float(tks[8]))
            id_list.append(tks[7])
            zs_list.append(float(tks[9]))
            break

print(len(embed_list))
print(len(label_list))
feature = np.array(embed_list)
print(feature.shape)

label_set_list = list(set(label_list))
labels = np.array([label_set_list.index(label) for label in label_list])
print(labels)
print(len(label_set_list), label_set_list)


conf_list_for_sort = copy.deepcopy(conf_list)
conf_list_for_sort.sort()
conf_med = conf_list_for_sort[len(conf_list_for_sort)//2]
conf_arr = np.array(conf_list)
conf_arr[conf_arr > conf_med] *= 5
conf_arr[conf_arr <= conf_med] /= 5


dist_list_for_sort = copy.deepcopy(dist_list)
dist_list_for_sort.sort()
dist_med = 1. - dist_list_for_sort[len(dist_list_for_sort)//2]
dist_arr = (1. - np.array(dist_list)).clip(0., 1.)
dist_arr[dist_arr > dist_med] *= 5
dist_arr[dist_arr <= dist_med] /= 5
print('dist_arr', dist_arr.shape)

zs_list_for_sort = copy.deepcopy(zs_list)
zs_list_for_sort.sort()
zs_med = zs_list_for_sort[len(zs_list_for_sort)//2]
zs_arr = np.array(zs_list)
zs_arr[zs_arr > zs_med] *= 5
zs_arr[zs_arr <= zs_med] /= 5

marker_list = []
for idx, gt in enumerate(label_list):
    if id_list[idx] == gt:
        marker_list.append('o')
    else:
        marker_list.append('*')
marker_arr = np.array(marker_list)
"""
iris = datasets.load_iris()

labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data,labels],axis=1)

feature = data[['Sepal length','Sepal width','Petal length','Petal width']]
"""

model = TSNE(learning_rate=1000)
transformed = model.fit_transform(feature)

xs = transformed[:, 0]
ys = transformed[:, 1]
mscatter(xs, ys, c=labels, s=conf_arr, m=marker_arr, cmap="nipy_spectral")
plt.gcf().set_dpi(600)
plt.show()

mscatter(xs, ys, c=labels, s=dist_arr, m=marker_arr, cmap="nipy_spectral")
plt.gcf().set_dpi(600)
plt.show()

mscatter(xs, ys, c=labels, s=zs_arr, m=marker_arr, cmap="nipy_spectral")
plt.gcf().set_dpi(600)
plt.show()