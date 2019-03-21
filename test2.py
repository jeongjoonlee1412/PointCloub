#!/usr/bin/python
# -*- coding: UTF-8 -*-
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import cv2
import csv
import skimage.data
import skimage.color

# a = np.array([[[10,10], [100,10], [100,100], [10,100]]], dtype=np.int32)
# b = np.array([[[100,100], [200,230], [150,200], [100,220]]], dtype=np.int32)
# im = np.zeros([240, 320], dtype = np.uint8)
# cv2.polylines(im, a, 1, 255,5)
# cv2.fillPoly(im, b, 255)
# cv2.imshow('test', im)
# cv2.waitKey(0)
# plt.imshow(im)
# plt.show()

a = [[10,10], [100,10], [100,100], [10,100]]
v_set = np.array(a)
Area = 0

v_dict = {}
no = 0
for v in v_set:
    v_dict[no] = v
    no += 1
for key in v_dict.keys():
    print(v_dict[key][0])
# get all vertex（顶点）
# get the crossover point（交叉点）
# plt.show()
# data = []
# kernel =np.array([[1, 0], [1, 1]])
# img = np.zeros((28, 28))
# csv_file = csv.reader(open('E:/robotStudy/digit-recognizer/test.csv', 'r'))
# for file in csv_file:
#     data.append(file)
# for i in range(1, 4):
#     for j in range(784):
#         img[j / 28, j % 28] = int(data[i][j])
#     plt.figure('dog')
#     print(img)
#    plt.show()
# img = Image.open('E:/1.jpg')
# img = np.array(img)

# plt.figure('dog')
# plt.imshow(img)
# plt.show()
