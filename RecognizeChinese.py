#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@author: Jeff LEE
@file: 图片识别.py
@time: 2018-07-20 10:59
@desc:
'''
from PIL import Image
import pytesseract

Image = Image.open('E:\\e_1.png')  # 打开图片
text = pytesseract.image_to_string(Image, lang='chi_sim')  # 使用简体中文解析图片
print(text)
