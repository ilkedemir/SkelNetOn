"""
Created on Tue Jul 31 11:16:54 2018

@author:  Ilke Demir, Camilla Hahn, Veronika Schulze

Create point clouds and labels for all shapes with either uniform, gaussian or beta distributed noise.
Change variable h for coarser or finer representation
"""

from __future__ import print_function
import numpy as np
from itertools import chain
from shapely.geometry import Point, Polygon
import os
import math

# Insert your local directory
dirskel = '/Users/camillahahn/Desktop/expskl/skl'

dircoords = '/Users/camillahahn/Desktop/expskl/bnd'

outdir = '/Users/camillahahn/Desktop/expskl/full'

os.makedirs(outdir)

np.random.seed()

# average distance along the axes between points; Change for coarser or finer representation
h = 1

# chose distribution of the noise; options: uniform, gauss or beta
noise = 'uniform'



def recursion(k, xmin, xmax, ymin, ymax, polygon, deep, h, noise='uniform'):
    """
    return: N x 2 array of points inside polygon
    """

    if k < deep:
        k += 1
        xmid = (xmax + xmin) / 2
        ymid = (ymax + ymin) / 2
        tmp1 = recursion(k, xmin, xmid, ymin, ymid, polygon, deep, h, noise)
        tmp2 = recursion(k, xmid, xmax, ymin, ymid, polygon, deep, h, noise)
        tmp3 = recursion(k, xmin, xmid, ymid, ymax, polygon, deep, h, noise)
        tmp4 = recursion(k, xmid, xmax, ymid, ymax, polygon, deep, h, noise)

        if tmp1.size:
            ret2 = np.vstack([tmp1, tmp2]) if tmp2.size else tmp1
            ret3 = np.vstack([ret2, tmp3]) if tmp3.size else ret2
            ret4 = np.vstack([ret3, tmp4]) if tmp4.size else ret3
        elif tmp2.size:
            ret3 = np.vstack([tmp2, tmp3]) if tmp3.size else tmp2
            ret4 = np.vstack([ret3, tmp4]) if tmp4.size else ret3
        elif tmp3.size:
            ret4 = np.vstack([tmp3, tmp4]) if tmp4.size else tmp3
        else:
            ret4 = tmp4

        return ret4
    else:
        p1 = Point(xmin, ymin)
        p2 = Point(xmin, ymax)
        p3 = Point(xmax, ymin)
        p4 = Point(xmax, ymax)
        pointlist = [p1, p2, p4, p3]
        pol = Polygon([[p.x, p.y] for p in pointlist])
        if polygon.intersects(pol):
            intersect = pol.intersection(polygon)

            x, y = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
            x = x.flatten()
            y = y.flatten()
            xy = np.array([x, y]).T
            if noise == 'uniform':
                xy_one = xy + np.random.rand(xy.shape[0], xy.shape[1]) * h
            elif noise == 'gauss':
                xy_one = xy + np.random.randn(xy.shape[0], xy.shape[1]) * h
            elif noise == 'beta':
                xy_one = xy + np.random.beta(2, 2, (xy.shape[0], xy.shape[1])) * h
            else:
                print('NOT DEFINED!')
                return

            mask_one = np.zeros(xy_one.shape[0], dtype=int)
            for k in range(xy_one.shape[0]):
                point = Point(xy_one[k, :])
                mask_one[k] = intersect.contains(point)

            mask_one = np.array(mask_one, dtype=bool)
            data_shape = xy_one[mask_one]

            return data_shape
        else:
            return np.array([])


os.chdir(dirskel)

for f in os.listdir(dirskel):
    # find file containing the skeleton points
    if not 'skelpoints' in f:
        continue
    # formating data
    data = open(f, 'r')
    data_skeleton = data.read()
    data_skeleton = data_skeleton.split('\n')
    data_skeleton = [t.split(' ') for t in data_skeleton]
    data_skeleton = [float(t) for t in list(chain.from_iterable(data_skeleton)) if t]
    data_skeleton_2 = np.array(data_skeleton).reshape(int(len(data_skeleton) / 2), 2)

    os.chdir(dircoords)

    file = f.replace('skelpoints', 'coords')
    data = open(file, 'r')
    data_shape = data.read()
    data_shape = data_shape.split('\n')
    data_shape = [t.split(' ') for t in data_shape]
    data_shape = [float(t) for t in list(chain.from_iterable(data_shape)) if t]
    data_shape_2 = np.array(data_shape).reshape(int(len(data_shape) / 2), 2)

    # produce point set filling the given shape
    xMin = data_shape_2[:, 0].min()
    xMax = data_shape_2[:, 0].max()
    yMin = data_shape_2[:, 1].min()
    yMax = data_shape_2[:, 1].max()
    polygon = Polygon(data_shape_2)

    data_full = recursion(1, xMin, xMax, yMin, yMax, polygon, 4, h, noise)
    data_full_skeleton = np.concatenate((data_full, data_skeleton_2), axis=0)

    os.chdir(outdir)

    # save points int .pts file
    np.savetxt(file.replace('.txt', '-full.pts'),
               data_full_skeleton, delimiter=' ')

    # determine skeleton points
    is_skeleton_point = np.ones(data_full.shape[0])
    for i in range(data_skeleton_2.shape[0]):
        distance = np.sqrt((data_skeleton_2[i, 0] -
                            data_full[:, 0]) ** 2 +
                           (data_skeleton_2[i, 1] - data_full[:, 1]) ** 2)
        is_skeleton_point[np.where(distance < math.sqrt(h))] = 2

    # save labels of points in .seg file
    is_skeleton = np.concatenate((is_skeleton_point, 2 * np.ones(data_skeleton_2.shape[0])))
    np.savetxt(file.replace('.txt', '-labels.seg'), is_skeleton, fmt='%i')

