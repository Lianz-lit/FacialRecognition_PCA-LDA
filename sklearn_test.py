#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pca 
@File    ：sklearn_test.py
@IDE     ：PyCharm 
@Author  ：Lianz
@Date    ：2023/4/2 20:02
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# 为方便显示图片，采用opencv读取、处理图片，用matplotlib显示图片
# 读取所有图片
root = '/Users/lit/Documents/博1/proj/[dataset]/orl_test'  # 根目录
listdir = os.listdir(root)
#listdir.sort(key=lambda s: int(s[1:]))  # listdir读取的目录列表无序，按数字大小进行排序

data = []

# 读取所有图片返回一个(m, n)数组，m为图片数量，n为图片一维size
for i in listdir:
    path = os.path.join(root, i)
    tmp_dir = os.listdir(path)
    tmp_dir.sort(key=lambda s: int(s.split('.')[0]))  # 按图片名称数字大小进行排序
    for j in tmp_dir:
        img_path = os.path.join(path, j)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('Failed to load image.')
        else:
            # print(img.shape)
            img = img.reshape(1, -1)
            data.append(img)
# 压缩后的全部图片 (400, 112*92)
data = np.array(data).reshape(len(data), -1)


# 展示其中n张
def show_img(figure_num, n, img_set):
    fig = plt.figure(figure_num)
    for i in range(n):
        tmp_img = img_set[i * 10].reshape(112, 92)
        plt.subplot(int(np.ceil(n/5)), 5, i + 1)
        plt.imshow(tmp_img, cmap='gray')
    plt.show()


show_img(1, 20, data)

pca = PCA(n_components=0.9)
new_data = pca.fit_transform(data)

print(data.shape)
print(new_data.shape)

data_inverse = pca.inverse_transform(new_data)

show_img(1, 20, data_inverse)




