#!/usr/bin/python
# -*- coding: UTF-8 -*-
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import csv
import skimage.data
import skimage.color

def convolution(data, kernel):
    data_row, data_col = np.shape(data)
    kernel_row, kernel_col = np.shape(kernel)
    n = data_col - kernel_col
    m = data_row - kernel_row
    state = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            temp = np.multiply(data[i:i+kernel_row,j:j+kernel_col], kernel)
            state[i][j] = temp.sum()
    return state


# a = np.arange(8/2, 16-8/2+1)
# t = np.uint16(a)
# print(t)
# print(a)

t = np.random.rand(3)
print(t)
# plt.imshow(img2)
# plt.imshow(img)
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
