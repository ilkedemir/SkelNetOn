
"""
Created on Wed Jan 23 09:11:33 2019

@author:  Ilke Demir, Camilla Hahn, Veronika Schulze

Upsample self generated data by a factor of 4
"""

from __future__ import print_function
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import os
import math

# Insert your local directory
dirname = '/Users/camillahahn/Desktop/expskl/full'

np.random.seed()

os.chdir(dirname)

for f in os.listdir(dirname):
    # find file containing full data
    if not 'full.pts' in f:
        continue

    # formating data
    data = open(f, 'r')
    data_full= data.read()
    data_full = data_full.split('\n')
    data_full = [t.split(' ') for t in data_full]
    data_full = [float(t) for t in list(chain.from_iterable(data_full)) if t]
    data_full_2 = np.array(data_full).reshape(int(len(data_full) / 2), 2)

    file = f.replace('full.pts', 'labels.seg')

    data = open(file, 'r')
    data_label = data.read()
    data_label = data_label.split('\n')
    if len(data_label) > data_full_2.shape[0]:
        del data_label[-1]
    data_label_2 = [list(map(int, x)) for x in data_label]
    data_label_2 = np.array(data_label_2)

    # generating the average distance between points to generate new h
    rand_int = np.random.random_integers(0, data_full_2.shape[0]-1, 50)
    dist = np.sqrt((data_full_2[rand_int, 0] - data_full_2[rand_int + 1, 0]) ** 2 +
                       (data_full_2[rand_int, 1] - data_full_2[rand_int, 1]) ** 2)
    dist_short = np.sort(dist)[5: 45]
    average_dist = np.sum(dist_short) / len(dist_short)
    h_new = average_dist / 2

    # upsampling
    sign1 = np.where(np.random.rand(data_full_2.shape[0]) < 0.5, -1, 1)
    sign2 = np.where(np.random.rand(data_full_2.shape[0]) < 0.5, -1, 1)
    new_data_1 = data_full_2.copy()
    new_data_2 = data_full_2.copy()
    new_data_1[:, 0] = new_data_1[:, 0] + h_new * sign1 + h_new * (np.random.rand(data_full_2.shape[0]) - 1/2)
    new_data_2[:, 1] = new_data_2[:, 1] + h_new * sign2 + h_new * (np.random.rand(data_full_2.shape[0]) - 1/2)
    new_data_3 = np.column_stack((new_data_1[:, 0], new_data_2[:, 1]))
    data_full_new = np.concatenate((data_full_2, new_data_1, new_data_2, new_data_3), axis=0)

    # update the labels
    index = np.array(np.where(data_label_2[:, 0] == 2)).flatten()
    skeleton = data_full_2[index, :]
    is_skeleton_point = np.ones(data_full_2.shape[0], dtype=int)
    for i in range(skeleton.shape[0]):
        distance = np.sqrt((skeleton[i, 0] -
                            data_full_2[:, 0]) ** 2 +
                           (skeleton[i, 1] - data_full_2[:, 1]) ** 2)
        is_skeleton_point[np.where(distance < math.sqrt(h_new))] = 2

    # save upsampled data
    np.savetxt(file.replace('labels', 'labels-up'), is_skeleton_point, delimiter=' ', fmt='%i')
    np.savetxt(file.replace('labels.seg', 'full-up.pts'), data_full_new)



