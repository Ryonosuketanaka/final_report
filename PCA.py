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


def pca_select(w):
    c_nume = np.sum(w)  # 累積寄与率の分子
    c_deno = 0  # 累積寄与率の分子
    c = c_nume / c_deno
    j = -1

    while c < 0.8 and j < np.size(w):  # 累積寄与率の基準を0.8
        j += 1
        c_deno += w[j]
        c = c_nume / c_deno

    return j


