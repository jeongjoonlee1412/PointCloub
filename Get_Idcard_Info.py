#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pytesseract
import cv2
import matplotlib.pyplot as plt
import dlib
import matplotlib.patches as mpatches
from skimage import io, draw, transform, color
import numpy as np
import pandas as pd
import re

detector = dlib.get_frontal_face_detector()
image = io.imread("E:\\img.jpg")
dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果
# 将识别的图像可视化
plt.figure()
ax = plt.subplot(111)
ax.imshow(image)
plt.axis("off")
for i, face in enumerate(dets):
    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    rect = mpatches.Rectangle((left, bottom), right - left, top - bottom,
                                  fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()

predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
detected_landmarks = predictor(image, dets[0]).parts()
landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
## 将眼睛位置可视化
# plt.figure()
# ax = plt.subplot(111)
# ax.imshow(image)
# plt.axis("off")
# plt.plot(landmarks[0:4,0],landmarks[0:4,1],'ro')
# for ii in np.arange(4):
#     plt.text(landmarks[ii,0]-10,landmarks[ii,1]-15,ii)
# plt.show()

## 计算眼睛的倾斜角度,逆时针角度
def twopointcor(point1,point2):
    """point1 = (x1,y1),point2 = (x2,y2)"""
    deltxy = point2 - point1
    corner = np.arctan(deltxy[1] / deltxy[0]) * 180 / np.pi
    return corner

## 计算多个角度求均值
corner10 =  twopointcor(landmarks[1,:],landmarks[0,:])
corner23 =  twopointcor(landmarks[3,:],landmarks[2,:])
corner20 =  twopointcor(landmarks[2,:],landmarks[0,:])
corner = np.mean([corner10,corner23,corner20])
# print(corner10)
# print(corner23)
# print(corner20)
# print(corner)

## 计算图像的身份证倾斜的角度
def IDcorner(landmarks):
    """landmarks:检测的人脸5个特征点
       经过测试使用第0个和第2个特征点计算角度较合适
    """
    corner20 =  twopointcor(landmarks[2,:],landmarks[0,:])
    corner = np.mean([corner20])
    return corner
corner = IDcorner(landmarks)
# print(corner)

## 将照片转正
def rotateIdcard(image):
    "image :需要处理的图像"
    ## 使用dlib.get_frontal_face_detector识别人脸
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 2) #使用detector进行人脸检测 dets为返回的结果
    ## 检测人脸的眼睛所在位置
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    detected_landmarks = predictor(image, dets[0]).parts()
    landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
    corner = IDcorner(landmarks)
    ## 旋转后的图像
    image2 = transform.rotate(image,corner,clip=False)
    image2 = np.uint8(image2*255)
    ## 旋转后人脸位置
    det = detector(image2, 2)
    return image2,det

## 转正身份证：
image = io.imread("img-0.png")
image2,dets = rotateIdcard(image)

## 可视化修正后的结果
plt.figure()
ax = plt.subplot(111)
# ax.imshow(image2)
plt.axis("off")
# 在图片中标注人脸，并显示
left = dets[0].left()
top = dets[0].top()
right = dets[0].right()
bottom = dets[0].bottom()
rect = mpatches.Rectangle((left,bottom), (right - left), (top - bottom),
                          fill=False, edgecolor='red', linewidth=1)
ax.add_patch(rect)

## 照片的位置（不怎么精确）
width = right - left
high = top - bottom
left2 = np.uint(left - 0.5*width)
bottom2 = np.uint(bottom + 0.5*width)
rect = mpatches.Rectangle((left2,bottom2), 1.8*width, 2.2*high,
                          fill=False, edgecolor='blue', linewidth=1)
ax.add_patch(rect)
plt.show()

## 身份证上人的照片
top2 = np.uint(bottom2+2.2*high)
right2 = np.uint(left2+1.8*width)
image3 = image2[top2:bottom2,left2:right2,:]
# plt.imshow(image3)
plt.axis("off")
plt.show()
# cv2.imshow('image3',image3)
# cv2.waitKey()

# ## 对图像进行处理，转化为灰度图像=>二值图像
# imagegray = cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
# cv2.imshow('imagegray',imagegray)
#
# cv2.waitKey()
# retval, imagebin = cv2.threshold(imagegray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# ## 将照片去除
# imagebin[0:bottom2,left2:-1] = 255
# # 高斯双边滤波
# img_bilateralFilter = cv2.bilateralFilter(imagebin, 40, 75, 75)
#
# cv2.imshow('img_bilateralFilter',img_bilateralFilter)
# cv2.waitKey()
# # plt.imshow(img_bilateralFilter,cmap=plt.cm.gray)
# #
# # plt.axis("off")
# # plt.show()


img=cv2.imread('img-0.png')  # 打开图片
gray=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)  # 灰度处理
# cv2.imshow('gray', gray)
retval, imagebin = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
## 将照片去除
imagebin[0:bottom2,left2:-1] = 255
img_bilateralFilter = cv2.bilateralFilter(imagebin, 40, 100, 100) # 高斯双边滤波

cv2.namedWindow("img_bilateralFilter", cv2.WINDOW_NORMAL)
cv2.imshow('img_bilateralFilter', img_bilateralFilter)

cv2.waitKey(0)
