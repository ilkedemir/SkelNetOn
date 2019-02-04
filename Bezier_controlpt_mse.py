#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:54:16 2019

@author: kathryn
"""

import numpy as np

def Bezier_controlpt_mse(true_vals, estim_vals):
    true_vals = np.asarray(true_vals)
    estim_vals = np.asarray(estim_vals)
    return  np.square(true_vals-estim_vals).mean(axis=None)