#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def picture_deal(image):
    """
    图片处理程序
    :param image: 图像
    :return: 处理好后的图像
    """
    kernel = np.ones((5, 5), np.uint8)
    # cv2.namedWindow(title)
    # cv2.imshow(title, img)
    # cv2.waitKey(500)

    # 图像灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(title, gray)
    # cv2.waitKey(500)

    # 灰度图像二值化
    retval, bina = cv2.threshold(gray, 190, 255, 0)
    # cv2.imshow(title, bina)
    # cv2.waitKey()

    # 开运算，去除大白斑周边的小白点
    # opening = cv2.morphologyEx(bina, cv2.MORPH_OPEN, kernel)
    # cv2.imshow(title, opening)
    # cv2.waitKey()

    # 闭运算，去除大白斑内的小黑点
    closing = cv2.morphologyEx(bina, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊滤波
    imblur = cv2.GaussianBlur(closing, (3, 3), 1.5)
    # cv2.imshow(title, bina)
    # cv2.waitKey()
    image = imblur

    # 边缘检测
    # canny = cv2.Canny(imblur, 50, 130, 3)
    # cv2.imshow(title, canny)
    # cv2.waitKey()

    # 显示图像来观察对比
    # htitch = np.hstack((bina, closing, imblur, canny))
    # cv2.imshow("test1", htitch)
    # cv2.waitKey()
    return image


def picture_divide():
    pass