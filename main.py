# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:54:24 2023

@author: ke3su
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PCA
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
plt.rcParams["font.family"] = ["Hiragino Maru Gothic Pro", "Yu Gothic"]


def load_text(fn):
    f = open(fn, "r")
    text = f.readlines()
    f.close()

    return text


def main():
    A = pd.read_csv("lec04_data_nagoya.csv", header=0, sep=",", encoding="shif\
t_jis")  # データファイル読み込み
    D = np.array(A, dtype="float")
    fn = r"lec04_data_sample_name_sjis.txt"
    sample_name_list = load_text(fn)  # 項目名のファイル読み込み

    z, w = PCA.pca_score(D)  # 主成分得点，固有値
    j = PCA.pca_select(w)

    Z = linkage(z[:, :j], metric="euclidean", method="ward")  # ウォード法
    d = dendrogram(Z, labels=sample_name_list)  # デンドログラム
    

if __name__ == "__main__":
    main()

