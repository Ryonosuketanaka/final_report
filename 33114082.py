# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:54:24 2023

@author: ke3su
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
plt.rcParams['font.family'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic']

def loadText(fn):     
    f = open(fn, 'r')
    
    text = f.readlines() # １行ごとリストになる
    
    f.close()
    return text

def PCA(D):     
    
    m = np.array(np.mean(D, axis=0)) #列ごとの平均
    s = np.std(D, axis=0, ddof=1) #列ごとの標準偏差を格納
    SD = (D-m)/s
    
    R = np.cov(SD,rowvar=False,ddof=1) #相関係数行列を計算
    w, v = np.linalg.eig(R) #固有値・固有ベクトルを計算
        
    sort_index = np.argsort(w)[::-1] #w,vを降順にソートさせる準備
    sort_w = w[sort_index] #wのソート
    sort_v = v[:,sort_index] #vのソート
    
    z = SD@sort_v

    return z,sort_w

if __name__ == "__main__":
    A = pd.read_csv("lec04_data_nagoya.csv", header = None, sep=",") #ファイル読み込み
    TD = np.array(A,dtype='float') #読みこんだcsvファイルを行列に

    z,w = PCA(TD) #6変数主成分得点
    
    fn = r"\Users\ke3su\Downloads\lec04_data_sample_name_sjis.txt"
    sample_name_list = loadText(fn) #区の名前
    
    nw = np.sum(w) #固有値の和
    sw = 0 #累積寄与率の分子
    c = sw/nw
    j = -1

    while c>0.8 :#累積寄与率を計算
        j += 1
        sw += w[j]
        c = sw/nw
 
     
    Z = linkage(z[:,:j], metric='euclidean', method='ward') #6変数第2主成分までのウォード法
    
    d = dendrogram(Z,labels = sample_name_list) #6変数のデンドログラム
    plt.figure()


