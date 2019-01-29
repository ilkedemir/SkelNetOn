
"""
Created on Wed Jan 23 09:11:33 2019

@author:  Ilke Demir, Camilla Hahn, Veronika Schulze

Downsample self generated data
"""

from __future__ import print_function
import numpy as np
from itertools import chain
import os

# Insert your local directory
dirname = '/Users/camillahahn/Desktop/expskl/full'

np.random.seed()

os.chdir(dirname)

# percentage of data to be kept; please change the value for your convenience
p = 0.5

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

    # downsampling
    N = int(data_full_2.shape[0] * p)
    index = np.random.random_integers(0, data_full_2.shape[0] - 1, N)
    data_n = data_full_2[index, :]
    label_n = data_label_2[index]

    # save downsampled data
    np.savetxt(file.replace('labels', 'labels-down'), label_n, delimiter=' ', fmt='%i')
    np.savetxt(file.replace('labels.seg', 'full-down.pts'), data_n)


