"""
Created on Tue Jul 31 11:16:54 2018

@author:  Ilke Demir, Camilla Hahn, Veronika Schulze

Adds noise to already existing point data. Choose between uniform, gaussian and beta distributed noise
"""

from __future__ import print_function
import numpy as np
from itertools import chain
import os
import math

# Insert your local directory
dirname = '/Users/camillahahn/Desktop/expskl/full'

np.random.seed()

# chose distribution of the noise; options: uniform, gauss or beta
noise = 'gauss'

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

    # getting the original skeleton points
    index = np.array(np.where(data_label_2[:, 0] == 2)).flatten()
    skeleton = data_full_2[index, :].copy()

    # getting the average distance between points
    rand_int = np.random.random_integers(0, data_full_2.shape[0] - 1, 50)
    dist = np.sqrt((data_full_2[rand_int, 0] - data_full_2[rand_int + 1, 0]) ** 2 +
                   (data_full_2[rand_int, 1] - data_full_2[rand_int, 1]) ** 2)
    dist_short = np.sort(dist)[10: 40]
    h = np.sum(dist_short) / (len(dist_short) * 2)

    #generating new pointset
    N = data_full_2.shape
    if noise == 'uniform':
        data_full_2 = data_full_2 + np.random.rand(N[0], N[1]) * h
    elif noise == 'gauss':
        data_full_2 = data_full_2 + np.random.randn(N[0], N[1]) * h
    elif noise == 'beta':
        data_full_2 = data_full_2 + np.random.beta(2, 2, (N[0], N[1])) * h
    else:
        print('NOT DEFINED!')
        break

    # save points int .pts file
    np.savetxt(file.replace('labels.seg', 'full-new.pts'), data_full_2, delimiter=' ')

    # getting the average distance between points
    rand_int = np.random.random_integers(0, data_full_2.shape[0] - 1, 50)
    dist = np.sqrt((data_full_2[rand_int, 0] - data_full_2[rand_int + 1, 0]) ** 2 +
                   (data_full_2[rand_int, 1] - data_full_2[rand_int, 1]) ** 2)
    dist_short = np.sort(dist)[10: 40]
    h = np.sum(dist_short) / (len(dist_short) * 2)

    # determine skeleton points
    is_skeleton_point = np.ones(data_full_2.shape[0], dtype=int)
    for i in range(skeleton.shape[0]):
        distance = np.sqrt((skeleton[i, 0] -
                            data_full_2[:, 0]) ** 2 +
                           (skeleton[i, 1] - data_full_2[:, 1]) ** 2)
        is_skeleton_point[np.where(distance < math.sqrt(h))] = 2

    # save labels of points in .seg file
    np.savetxt(file.replace('labels', 'labels-new'), is_skeleton_point, fmt='%i')


