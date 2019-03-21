#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from time import time
import sys
import json
import base64
from wangyx import getAreaBrightness


def clip(src, ploy):
    """
    clip the src to dst with ploy as mask.
    :param src: 图片矩阵img
    :param ploy: 多边形区域顶点坐标矩阵
    :return:
    """
    t1 = time()

    # 在空白图片上依次画出每个床位躺坐站的检测区域（用途未知）
    empimg = np.zeros(src.shape, np.uint8)
    mask = empimg.copy()
    # t2 = time()
    cv2.fillPoly(mask, ploy, (255, 255, 255))
    # cv2.imshow('test', mask)
    # cv2.waitKey()

    # 用途未知
    # t3 = time()
    area = int(0)
    b = float(0)  # sum bright
    maxb = int(0)  # max bright
    h, w = src.shape[0], src.shape[1]
    # lx, ly, rx, ry = 0, 0, w-1, h-1
    # for e in ploy[0]:
    #     [x, y] = e
    #     lx = x if x < lx else lx
    #     rx = x if x > rx else rx
    #     ly = y if y < ly else ly
    #     ry = y if y > ry else ry
    
    maxp = [0, 0]
    # ==============================================
    # switch algorithm (交换算法)
    # ==============================================
    # test_wangyx = True
    # if test_wangyx:
    #     area, avgb = getAreaBrightness(ploy, src)
    #     b = area*avgb
    # elif len(src.shape) == 2:  # gray image
    #     for i in range(ly, ry):
    #         for j in range(lx, rx):
    #             if mask[i][j] == 0:
    #                 continue
    #             tmb = src[i][j]
    #             mask[i][j] = tmb
    #             area += 1
    #             b += tmb
    #             maxb = tmb if tmb > maxb else maxb
    #             if tmb == maxb:
    #                 maxp = [j, i]
    # else:  # color image
    #     for i in range(src.shape[0]-1):
    #         for j in range(src.shape[1]-1):
    #             if mask[i][j][0] == 0:
    #                 continue
    #             tmb = max(max(src[i][j][0], src[i][j][1]), src[i][j][2])
    #             mask[i][j] = src[i][j]
    #             area +=1
    #             b += tmb
    #             maxb = tmb if tmb > maxb else maxb
    #             if tmb == maxb:
    #                 maxp = [j, i]
    # t4 = time()
    #
    # return int(b/max(1, area)), maxb, maxp, area, int(b), 1000*(t4-t1)
    getAreaBrightness(ploy, src)
    # b = area*avgb
    if len(src.shape) == 2:  # gray image
        for i in range(h):
            for j in range(w):
                if mask[i][j] == 0:
                    continue
                tmb = src[i][j]
                mask[i][j] = tmb
                area += 1
                b += tmb
                maxb = tmb if tmb > maxb else maxb
                if tmb == maxb:
                    maxp = [j, i]
    else:  # color image
        for i in range(src.shape[0]-1):
            for j in range(src.shape[1]-1):
                if mask[i][j][0] == 0:
                    continue
                tmb = max(max(src[i][j][0], src[i][j][1]), src[i][j][2])
                mask[i][j] = src[i][j]
                area +=1
                b += tmb
                maxb = tmb if tmb > maxb else maxb
                if tmb == maxb:
                    maxp = [j, i]

    t4 = time()

    return int(b/max(1, area)), maxb, maxp, area, int(b), 1000*(t4-t1)


def getperson(a, b, c, d, A, B, C, D):
    '''
    (a,c),(b,d) is the front rectangle
    (A,C),(B,D) if the background rectangle
    '''
    return [[b,c],[b,d],[B,D],[B,C],[A,C],[a,c]]
    #return [[a,c],[b,c], [b,d], [B,D], [B,C],[A,C]]


# 没有引用
def findperson(plts,bx0,bx1,bx2,bx3,b0,b1,b2,b3, t):
    '''
    xx is the front rectangle
    (A,C),(B,D) if the background rectangle
    '''
    pass
    # return [[b,c], [b,d], [B,D], [B,C],[b,c],[a,c],[A,C],[B,C]]


# 没有引用
def getPloy(xx, b1, b2, b3):
    poly, cl = [], (255,128,0)
    poly.append(getBed(img, [xx[0],xx[1]], b1[:4], cl))
    poly.append(getBed(img, [xx[0],xx[1]], b1[3:], cl))
    poly.append(getBed(img, [xx[10],xx[11]], b1[:4], cl))
    poly.append(getBed(img, [xx[10],xx[11]], b1[3:], cl))
    
    poly.append(getBed(img, [xx[2],xx[3]], b2[:4], cl))
    poly.append(getBed(img, [xx[2],xx[3]], b2[3:], cl))
    poly.append(getBed(img, [xx[8],xx[9]], b2[:4], cl))
    poly.append(getBed(img, [xx[8],xx[9]], b2[3:], cl))
    
    poly.append(getBed(img, [xx[4],xx[5]], b3[:4], cl))
    poly.append(getBed(img, [xx[4],xx[5]], b3[3:], cl))
    poly.append(getBed(img, [xx[6],xx[7]], b3[:4], cl))
    poly.append(getBed(img, [xx[6],xx[7]], b3[3:], cl))
    return poly


def getBed(img, x, yy, cl=(255, 128, 0)):
    '''
    '''
    ret1 = []
    ret2 = []
    y0 = -1
    for y in yy:
        pl0=(x[0],y0)
        pr0=(x[1],y0)
        if y0 == -1:
            y0 = y
            ret1.append([x[0],y0])
            ret2.append([x[1],y0])
            continue
        pl1=(x[0],y)
        pr1=(x[1],y)
        cv2.line(img, pl0, pl1, cl)
        cv2.line(img, pr0, pr1, cl)
        cv2.line(img, pl1, pr1, cl)
        cv2.circle(img, pl0, 5, cl)
        cv2.circle(img, pr0, 5, cl)
        cv2.circle(img, pl1, 8, cl)
        cv2.circle(img, pr1, 8, cl)
        y0 = y
        ret1.append([x[0],y])
        ret2.append([x[1],y])
    ret2.reverse()
    return ret1+ret2


# 没有引用
def lineBed(img, x, yy, cl=(255, 128, 0)):
    '''
    '''
    y0 = -1
    for y in yy:
        if y0 == -1:
            y0 = y
            continue
        pll0=(x[0],y0)
        pll1=(x[0],y)
        plr0=(x[1],y0)
        plr1=(x[1],y)
        prl0=(x[2],y0)
        prl1=(x[2],y)
        prr0=(x[3],y0)
        prr1=(x[3],y)
        #cv2.line(img, pll0, pll1, cl)
        #cv2.line(img, plr0, plr1, cl)
        #cv2.line(img, prr0, prr1, cl)
        #cv2.line(img, prl0, prl1, cl)
        #cv2.line(img, pll1, plr1, cl)
        #cv2.line(img, prl1, prr1, cl)
        #cv2.circle(img, pll0, 5, cl)
        #cv2.circle(img, plr0, 5, cl)
        #cv2.circle(img, prr0, 5, cl)
        #cv2.circle(img, prl0, 5, cl)
        cv2.circle(img, pll1, 8, cl)
        cv2.circle(img, plr1, 8, cl)
        cv2.circle(img, prr1, 8, cl)
        cv2.circle(img, prl1, 8, cl)
        y0 = y
        

def main_proc(file_src, param, threshold):
    img = cv2.imread(file_src)
    title = 'cv2py'
    kernel = np.ones((5, 5), np.uint8)
    # cv2.namedWindow(title)
    # cv2.imshow(title, img)
    # cv2.waitKey(500)

    # 图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    img = imblur

    # 边缘检测
    # canny = cv2.Canny(imblur, 50, 130, 3)
    # cv2.imshow(title, canny)
    # cv2.waitKey()

    # 显示图像来观察对比
    # htitch = np.hstack((bina, closing, imblur, canny))
    # cv2.imshow("test1", htitch)
    # cv2.waitKey()

    # 获取轮廓，计算轮廓的相关信息（用途暂时未知）
    # cts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for i in range(0, len(cts)):
    #     br = cv2.boundingRect(cts[i])    # 计算轮廓的垂直边界最小矩形，返回x，y，w，h
    #     ar = cv2.contourArea(cts[i])     # 计算轮廓面积
    #     ac = cv2.arcLength(cts[i], False)  # 检测物体的周长
    #     mu = cv2.moments(cts[i], False)   # 计算图像中的中心矩(最高到三阶)
    #     # if len(cts[i]) > 200:
    #     #     #print ((br, ar, ac))
    #     #     #print (mu)
    #     #     #print ('\n')
    #     #     pass

    h, w = img.shape[0], img.shape[1]
    cen = (w//2, h//2)

    xx, b0, b1, b2, b3 = param
    xx = [int(e*w) for e in xx]  # [0,0.204166667,0.159027778,0.305555556,0.24375,0.355555556,0.644444444,0.75625,0.694444444,0.840972222,0.795833333,1]
    b0 = [int(e*h) for e in b0]  # [0.0        ,0.257986111,0.387986111,0.664236111,0.938541667, 1.0,        1.0]
    b1 = [int(e*h) for e in b1]  # [0.0        ,0.243055556,0.407986111,0.564236111,0.838541667, 1.0,        1.0]
    b2 = [int(e*h) for e in b2]  # [0.118055556,0.329861111,0.421875,   0.522569444,0.71875,     0.777777778,0.9375]
    b3 = [int(e*h) for e in b3]  # [0.239583333,0.375661376,0.439236111,0.5,        0.659722222, 0.708333333,0.777777778]

    # lineBed(img, [xx[0],xx[1],xx[10],xx[11]], b1, (255,128,0))
    # lineBed(img, [xx[2],xx[3],xx[8] ,xx[9]],  b2, (0,128,255))
    # lineBed(img, [xx[4],xx[5],xx[6] ,xx[7]],  b3, (128,225,128))
    #
    # poly, cl = [], (255,128,0)
    # poly.append(getBed(img, [xx[0],xx[1]], b1[:4], cl))
    # poly.append(getBed(img, [xx[0],xx[1]], b1[3:], cl))
    # poly.append(getBed(img, [xx[10],xx[11]], b1[:4], cl))
    # poly.append(getBed(img, [xx[10],xx[11]], b1[3:], cl))
    #
    # poly.append(getBed(img, [xx[2],xx[3]], b2[:4], cl))
    # poly.append(getBed(img, [xx[2],xx[3]], b2[3:], cl))
    # poly.append(getBed(img, [xx[8],xx[9]], b2[:4], cl))
    # poly.append(getBed(img, [xx[8],xx[9]], b2[3:], cl))
    #
    # poly.append(getBed(img, [xx[4],xx[5]], b3[:4], cl))
    # poly.append(getBed(img, [xx[4],xx[5]], b3[3:], cl))
    # poly.append(getBed(img, [xx[6],xx[7]], b3[:4], cl))
    # poly.append(getBed(img, [xx[6],xx[7]], b3[3:], cl))

    x = xx
    bx0 = [0, 0, w-1, w-1]
    bx1 = [x[0], x[1], x[11], x[10]]
    bx2 = [x[2], x[3], x[9], x[8]]
    bx3 = [x[4], x[5], x[7], x[6]]
    lie_plts, sit_plts, std_plts = [], [], []
    lie_plts.append(getperson(bx0[0], bx0[1], b0[2], b0[3], bx1[0], bx1[1], b1[2], b1[3]))
    lie_plts.append(getperson(bx0[2], bx0[3], b0[2], b0[3], bx1[2], bx1[3], b1[2], b1[3]))
    lie_plts.append(getperson(bx1[0], bx1[1], b1[2], b1[3], bx2[0], bx2[1], b2[2], b2[3]))
    lie_plts.append(getperson(bx1[2], bx1[3], b1[2], b1[3], bx2[2], bx2[3], b2[2], b2[3]))
    lie_plts.append(getperson(bx2[0], bx2[1], b2[2], b2[3], bx3[0], bx3[1], b3[2], b3[3]))
    lie_plts.append(getperson(bx2[2], bx2[3], b2[2], b2[3], bx3[2], bx3[3], b3[2], b3[3]))
    lie_plts.append(getperson(bx0[0], bx0[1], b0[4], b0[5], bx1[0], bx1[1], b1[4], b1[5]))
    lie_plts.append(getperson(bx0[2], bx0[3], b0[4], b0[5], bx1[2], bx1[3], b1[4], b1[5]))
    lie_plts.append(getperson(bx1[0], bx1[1], b1[4], b1[5], bx2[0], bx2[1], b2[4], b2[5]))
    lie_plts.append(getperson(bx1[2], bx1[3], b1[4], b1[5], bx2[2], bx2[3], b2[4], b2[5]))
    lie_plts.append(getperson(bx2[0], bx2[1], b2[4], b2[5], bx3[0], bx3[1], b3[4], b3[5]))
    lie_plts.append(getperson(bx2[2], bx2[3], b2[4], b2[5], bx3[2], bx3[3], b3[4], b3[5]))

    sit_plts.append(getperson(bx0[0], bx0[1], b0[1], b0[2], bx1[0], bx1[1], b1[1], b1[2]))
    sit_plts.append(getperson(bx0[2], bx0[3], b0[1], b0[2], bx1[2], bx1[3], b1[1], b1[2]))
    sit_plts.append(getperson(bx1[0], bx1[1], b1[1], b1[2], bx2[0], bx2[1], b2[1], b2[2]))
    sit_plts.append(getperson(bx1[2], bx1[3], b1[1], b1[2], bx2[2], bx2[3], b2[1], b2[2]))
    sit_plts.append(getperson(bx2[0], bx2[1], b2[1], b2[2], bx3[0], bx3[1], b3[1], b3[2]))
    sit_plts.append(getperson(bx2[2], bx2[3], b2[1], b2[2], bx3[2], bx3[3], b3[1], b3[2]))
    sit_plts.append(getperson(bx0[0], bx0[1], b0[3], b0[4], bx1[0], bx1[1], b1[3], b1[4]))
    sit_plts.append(getperson(bx0[2], bx0[3], b0[3], b0[4], bx1[2], bx1[3], b1[3], b1[4]))
    sit_plts.append(getperson(bx1[0], bx1[1], b1[3], b1[4], bx2[0], bx2[1], b2[3], b2[4]))
    sit_plts.append(getperson(bx1[2], bx1[3], b1[3], b1[4], bx2[2], bx2[3], b2[3], b2[4]))
    sit_plts.append(getperson(bx2[0], bx2[1], b2[3], b2[4], bx3[0], bx3[1], b3[3], b3[4]))
    sit_plts.append(getperson(bx2[2], bx2[3], b2[3], b2[4], bx3[2], bx3[3], b3[3], b3[4]))

    std_plts.append(getperson(bx0[0], bx0[1], b0[0], b0[1], bx1[0], bx1[1], b1[0], b1[1]))
    std_plts.append(getperson(bx0[2], bx0[3], b0[0], b0[1], bx1[2], bx1[3], b1[0], b1[1]))
    std_plts.append(getperson(bx1[0], bx1[1], b1[0], b1[1], bx2[0], bx2[1], b2[0], b2[1]))
    std_plts.append(getperson(bx1[2], bx1[3], b1[0], b1[1], bx2[2], bx2[3], b2[0], b2[1]))
    std_plts.append(getperson(bx2[0], bx2[1], b2[0], b2[1], bx3[0], bx3[1], b3[0], b3[1]))
    std_plts.append(getperson(bx2[2], bx2[3], b2[0], b2[1], bx3[2], bx3[3], b3[0], b3[1]))

    output = []

    for e in lie_plts:
        # 画多边形
        cv2.polylines(img, [np.array(e)], True, (255, 0, 0))
        # cv2.imshow('test', img)
        # cv2.waitKey()
        output.append((e, 'lie', clip(img, [np.array(e)])))
    # cv2.imshow(title, img)
    # cv2.waitKey(0)

    for e in sit_plts:
        cv2.polylines(img, [np.array(e)], True, (0, 255, 0))
        # cv2.imshow('test', img)
        # cv2.waitKey(1000)
        output.append((e, 'sit', clip(img, [np.array(e)])))
    # cv2.imshow(title, img)
    # cv2.waitKey(0)

    for e in std_plts:
        # cv2.polylines(img, [np.array(e)], True, (np.random.randint(255),
        # np.random.randint(255),np.random.randint(255)))
        cv2.polylines(img, [np.array(e)], True, (0, 0, 255))
        # cv2.imshow('test', img)
        # cv2.waitKey(1000)
        output.append((e, 'std', clip(img, [np.array(e)])))
    # cv2.imshow(title, img)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
    return img, output


if __name__ == "__main__":
    file_src = '1.png'
    dstfile = 'output.txt'
    threshold = 112
    param =[
        [0,0.204166667,0.159027778,0.305555556,0.24375,0.355555556,0.644444444,0.75625,0.694444444,0.840972222,0.795833333,1],
        [0.0        ,0.257986111,0.387986111,0.664236111,0.938541667, 1.0,        1.0],
        [0.0        ,0.243055556,0.407986111,0.564236111,0.838541667, 1.0,        1.0],
        [0.118055556,0.329861111,0.421875,   0.522569444,0.71875,     0.777777778,0.9375],
        [0.239583333,0.375661376,0.439236111,0.5,        0.659722222, 0.708333333,0.777777778]]

    person=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    position=['left_up', 'right_up', 'left_down', 'right_down']
    bed = ['No.1_left_up_','No.1_left_down_','No.1_right_up_','No.1_right_down_',
           'No.2_left_up_','No.2_left_down_','No.2_right_up_','No.2_right_down_',
           'No.3_left_up_','No.3_left_down_','No.3_right_up_','No.3_right_down_',
           'No.1_left_up_','No.1_left_down_','No.1_right_up_','No.1_right_down_',
           'No.2_left_up_','No.2_left_down_','No.2_right_up_','No.2_right_down_',
           'No.3_left_up_','No.3_left_down_','No.3_right_up_','No.3_right_down_',
           'No.1_left_up_','No.1_right_up_','No.2_left_up_','No.2_right_up_','No.3_left_up_','No.3_right_up_']
    # 命令行输入参数
    # if len(sys.argv)>1:
    #     file_src = sys.argv[1]
    # if len(sys.argv)>2:
    #     dstfile = sys.argv[2]
    # if len(sys.argv)>3:
    #     threshold = int(sys.argv[3])
    # if len(sys.argv)>4:
    #     file_param = sys.argv[4]
    #     with open(file_param, 'r') as f:
    #         param = json.load(f)
    file_bin = sys.argv[0]
        
    # begin main process
    img, output = main_proc(file_src, param, threshold)

    dic_out = {"img": "", "beds": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]}
    f1 = sys.argv[0]
    # pars
    for i, e in enumerate(output):  # 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        pplt, nm, (avb, maxb, maxp, area, bw, tmsp) = e
        info1 = '_______body_is_'+nm
        if avb < threshold:
            info1 = 'no_body'
        info = bed[i]+info1
        ostr = '"name":"{6}","position":"{7}",\n"avg_bright":{0},\n"max_bright":{1},\n"max_point":{2},' \
               '\n"area":{3},\n"bright_weaght":{4}, \n"time_span":{5}\n'.format(avb, maxb, maxp, area, bw, tmsp, info, pplt )
        if i < 12:
            person[i] = ostr
            dic_out['beds'][i % 12]['NO'] = str(i+1)
            # status 是1，则表示该床位有人
            dic_out['beds'][i % 12]['status'] = 0 if avb < threshold else 1
        elif avb > threshold:
            person[i % 12] = ostr
            dic_out['beds'][i % 12]['status'] = 2

    cv2.imwrite("im1.png", img)
    b64 = base64.b64encode(open("im1.png","rb").read()).decode('utf-8')
    dic_out['img'] = b64

    json_str = json.dumps(dic_out)
    # write to output file.
    of = open(dstfile, 'w')
    if None != of:
        try:
            of.write(json_str)
            for e in person:
                print (e)
                of.write(''.join(e))
            of.close()
        finally:
            pass
