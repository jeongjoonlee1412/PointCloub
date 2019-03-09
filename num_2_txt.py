#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PIL import Image
import matplotlib.pylab as plt
import numpy as np


def picTo01(filename):
    """
    将图片转换为32*32像素的文件，用0、1表示
    :param filename:
    :return:
    """
    # 打开图片
    img = Image.open(filename).convert('RGBA')

    # 得到图片的像素值
    raw_data = img.load()
    print(img.size[1])
    print(img.size[0])
    print raw_data[0, 0]

    # 将其降噪转化为黑白两色
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][0] < 90:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][1] < 136:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][2] > 0:
                raw_data[x, y] = (255, 255, 255, 255)

    # 设置为32*32的大小
    img = img.resize((32, 32), Image.BICUBIC)

    # 进行保存，方便查看
    img.save('E:\\test.png')

    # 得到像素值组，为（32,32,4）
    array = plt.array(img)

    # 按照公式将其转换为01，公式：0.299*R + 0.587*G + 0.114*B
    gray_array = np.zeros((32, 32))

    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            gray = 0.299 * array[x][y][0] + 0.587 * array[x][y][1] + 0.114 * array[x][y][2]

            # 设置一个阙值，记为0
            if gray == 255:
                gray_array[x][y] = 0
            else:
                # 否则认为是黑色，全记为1
                gray_array[x][y] = 1

    # 得到对应名称的TXT文件
    name01 = filename.split('.')[0]
    name01 = name01 + '.txt'

    # 保存到文件
    np.savetxt(name01, gray_array, fmt='%d', delimiter='')


if __name__ == '__main__':
    picTo01('E:\\nearest.png')
