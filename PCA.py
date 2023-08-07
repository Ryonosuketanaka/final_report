# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:54:03 2023

@author: ke3su
"""
import numpy as np


def pca(D):
    m = np.array(np.mean(D, axis=0))
    s = np.std(D, axis=0, ddof=1)
    D_norm = (D - m) / s

    R = np.cov(D_norm, rowvar=False, ddof=1)
    w, v = np.linalg.eig(R)

    sort_index = np.argsort(w)[::-1]
    sort_w = w[sort_index]
    sort_v = v[:, sort_index]

    return sort_w, sort_v


def pca_score(D):
    w, v = pca(D)

    m = np.array(np.mean(D, axis=0))
    s = np.std(D, axis=0, ddof=1)
    D_norm = (D - m) / s

    z = D_norm @ v

    return z, w





