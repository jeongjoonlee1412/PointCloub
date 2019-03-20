#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import urllib
#import urllib2
import scipy
import cv2
import numpy as np


# 计算 p1和p2两点之间的距离
def distance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


# 判断 tri_v中是否有三个元素，无则返回false；有则判断三个点是否能构成三角形，否返回false，是返回TRUE
def isTriangle(tri_v):
    if( not (len(tri_v) == 3) ):
        return False
    return (distance(tri_v[0], tri_v[1]) + distance(tri_v[1], tri_v[2]) >
            distance(tri_v[0], tri_v[2])) and  (distance(tri_v[1], tri_v[2]) + distance(tri_v[2], tri_v[0]) >
            distance(tri_v[0], tri_v[1])) and  (distance(tri_v[2], tri_v[0]) + distance(tri_v[0], tri_v[1]) >
            distance(tri_v[1], tri_v[2]))


# def get
# 暂时不明白
def mutiPly(v1, v2, p):
    return ((p[1] - v1[1])*(v2[0] - v1[0]) - (p[0] - v1[0])*(v2[1] - v1[1]))


# 判断x是否在x1和x2之间
def between(x1, x2, x):
    l = sorted([x1, x2])  # 将x1, x2进行由小到大排序
    if( x>l[0] and  x<l[1] ):
        return True
    else:
        return False


'''
功能：p1点和p2点构成一条直线，p3点和p4点构成一条直线，求出两条直线的交叉点
输入：p1, p2, p3, p4四个点位坐标
输出：situation1：[False, 0]：四个点位不存在交点，返回一个列表[];
      situation1：[True, [p2[0], (float(p4[1]-p3[1]) / (p4[0] - p3[0]))*(p2[0] - p3[0]) + p3[1]]]:存在交点，并返回交点坐标
'''
def getCrossoverPoint(p1, p2, p3, p4):
    if( ( (p2[0] - p1[0]==0) and (p4[0] - p3[0]==0)) ):#四个点构成矩形
        return [False, 0]
    if( float(p2[1] - p1[1]) / (p2[0] - p1[0]) == (float(p4[1] - p3[1]) / (p4[0] - p3[0]) ) ):#四个点构成平行四边形
        return [False, 0]
    if( p2[0] - p1[0] == 0 and not (p4[0] - p3[0] == 0) ):
        return [True, [p2[0], (float(p4[1]-p3[1]) / (p4[0] - p3[0]))*(p2[0] - p3[0]) + p3[1]]]
    elif( p4[0] - p3[0] == 0 and not (p2[0] - p1[0] == 0) ):
        return [True, [p3[0], (float(p2[1]-p1[1]) / (p2[0] - p1[0]))*(p3[0] - p1[0]) + p1[1]]]
    else:
        k1 = float(p2[1] - p1[1]) / (p2[0] - p1[0])   #/为浮点数除法；//为取整除法；%为取余除法
        k2 = float(p4[1] - p3[1]) / (p4[0] - p3[0])
        b1 = p2[1] - float(k1*p2[0])
        b2 = p4[1] - float(k2*p4[0])
        x = float(b2 - b1) / (k1 - k2)
        y = float(k1*b2 - b1*k2) / (k1 - k2)
        if( between(p1[0],p2[0],x) and between(p3[0], p4[0], x) ):
            return [True, [x, y]]
        return [False, 0]

'''def insideConvexPolygon(v_dict, v):
    x = 0
    y = 0
    for key in v_dict:
        x += v_dict[key][0]
        y += v_dict[key][0]
    ave_x = x / len(v_dict.keys())
    ave_y = y / len(v_dict.keys())
    for key in v_dict:
        v1 = v_dict[key]
        v2 = v_dict[(key + 1)%len(v_dict.keys())]
        if( mutiPly(v1, v2, [ave_x,ave_y]) * mutiPly(v1, v2, v) < 0 )
            break
    return (key+1) == len(v_dict.keys())'''


'''
功能：分割多边形成为三角形，并获得三角形顶点坐标
输入：v_dict:多边形点集
输出：[True, tri_v]：三角形点集
'''
def partPolygon(v_dict):
    #get a convex vertex(凸面顶点)
    #vertex with smallest x must be convex
    smallest_x = float("inf")
    for key in v_dict.keys():
        if(smallest_x > v_dict[key][0]):
            smallest_x = v_dict[key][0]
            cov_key = key
    
    #part（分开） the polygon（多边形） with a tringle（支撑杆）
    v_len = len(v_dict.keys())
    pre_key = (cov_key - 1 + v_len)%v_len
    while( v_dict.get(pre_key, 0) == 0 or v_dict.get(pre_key, 0) == v_dict[cov_key] ):
        pre_key = (pre_key - 1 + v_len)%v_len
        #print pre_key
    next_key = (cov_key + 1 + v_len)%v_len
    while( v_dict.get(next_key, 0) == 0 or v_dict.get(next_key, 0) == v_dict[cov_key] ):
        next_key =(next_key + 1 + v_len)%v_len
    tri_v = [ v_dict[pre_key], v_dict[cov_key], v_dict[next_key] ]
    if( isTriangle(tri_v) ):
        return [True, tri_v]
    return [False, 0]



'''def partPolygon(v_dict, ):
    smallest_x = float("inf")
    cov_key = 0
    for key in v_dict.keys():
        if(smallest_x > v_dict[key][0]):
            smallest_x = v_dict[key][0])
            cov_key = key

    key_len = len(v_dict.keys())
    bin = False
    tri_v = [ v_dict[(cov_key - 1 + key_len)%key_len], v_dict[cov_key], v_dict[(cov_key + 1 + key_len)%key_len] ]  
    #key_len = len(v_dict.keys())
    for key in v_dict.keys():
        if( key = cov_key ):
            continue
        if( key = cov_key - 1 + len(v_dict.keys()) ):
            continue 
        if( key = cov_key + 1 + len(v_dict.keys()) ):
            continue
        if( !insideConvexPolygon(tri_v, v_dict[key]) ):
            continue
        bin = True'''

'''
功能：知道三角形三个顶点坐标，通过海伦定理求出三角形面积
输入：l：三角形三个顶点坐标；
输出：np.sqrt(p*(p-a)*(p-b)*(p-c))：三角形面积
'''
def getTriArea(l):
    #Helen theorem #海伦定理
    a = np.sqrt(np.square(l[0][0] - l[1][0]) + np.square(l[0][1] - l[1][1]))
    b = np.sqrt(np.square(l[1][0] - l[2][0]) + np.square(l[1][1] - l[2][1]))
    c = np.sqrt(np.square(l[2][0] - l[0][0]) + np.square(l[2][1] - l[0][1]))
    p = (a + b + c) / 2
    return np.sqrt(p*(p-a)*(p-b)*(p-c))

#tri_v:point list
#p_b:matrix of brightness
#return total brightness and number of point
'''
功能：获得点列表内的总的亮度（像素值）和点构成三角形内的总点数
输入：tri_v：构成三角形的点列表； p_b：像素矩阵
输出：[num_p, brightness]：总点数和总亮度
'''
def getTriBrightness(tri_v, p_b):
    x_l = []
    y_l = []
    num_p = 0
    brightness = 0
    for v in tri_v:
        x_l.append(v[0])
        y_l.append(v[1])
    x_l.sort()
    y_l.sort()
    for i in range(x_l[0], x_l[2]+1):
        for j in range(y_l[0], y_l[2]+1):
            #if the point in the triganle
            #judge the point in the left or right of the line
            #点是否在三角形内，判断点在线段的左侧还是右侧
            if( mutiPly(tri_v[0], tri_v[1], [i, j]) * mutiPly(tri_v[1], tri_v[2], [i, j]) >= 0
                and mutiPly(tri_v[0], tri_v[1], [i,j]) * mutiPly(tri_v[0], tri_v[2], [i,j]) >= 0   ):
                num_p += 1
                brightness += p_b[i][j]
    return [num_p, brightness]
            
#v_set:list,vertex of the convex multi deformation
#url:url of the picture
#path:path to save the picture 
#size:list,size of picture
def getAreaBrightnessx(v_set, url):
    p = urllib.urlopen(url)
    image = np.asarray(bytearray(p.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.GRAYSCALE)
    return getAreaBrightness(v_set, image)

'''
功能：获得区域面积和区域平均亮度（像素值）
输入：v_set：构成指定区域的点集 image：原图
输出：[Area, Ave_Brightness]：区域面积和区域平均亮度（像素值）
'''
def getAreaBrightness(v_set, image):
    Area = 0
    Brightness = 0
    num_p = 0
    
    #get all vertex（顶点）
    #get the crossoverpoint（交叉点）
    v_tmp = v_set
    for i in range(0, len(v_tmp)-1):
        for j in range(i+1, len(v_tmp)-1):
            res = getCrossoverPoint(v_tmp[i], v_tmp[i+1], v_tmp[j], v_tmp[j+1])#获得交点坐标
            print (res[0])
            if( res[0] ):
                v_set.insert(i+1, res[1])
                v_set.insert(j+2, res[1])

    #load the picture

    vertex = []
    v_dict = {}
    no = 0
    for v in v_set:
        v_dict[no] = v
        no += 1
    
    while( len(v_dict.keys()) > 3 ):
        tri = partPolygon(v_dict)#分割多边形成为三角形，并获得三角形顶点坐标
        if( tri[0] ):
            Area += getTriArea(tri[1])#获得三角形面积
            num_p += getTriBrightness(tri[1], image)[0]#获得三角形区域亮度（像素值）
            Brightness += getAreaBrightness(tri[1], image)[1]#获得点列表内的总的亮度（像素值）和点构成三角形内的总点数

            for key in v_dict.keys():
                if( v_dict.get(key, 0) == tri[1][1] ):
                    v_dict.pop(key)

            tmp_no = 0
            while( v_dict.get(tmp_no, 0) == 0 ):
                tmp_no += 1
            if( not (v_dict.get(tmp_no, 0) == v_dict.get(tmp_no + len(v_dict.keys()) , 0)) ):
                v_dict[tmp_no + len(v_dict.keys()) ] = v_dict[tmp_no]
            if( v_dict[tmp_no + 1] == v_dict[tmp_no + len(v_dict.keys()) - 1] ):
                v_dict.pop(tmp_no)
                v_dict.pop(tmp_no + len(v_dict.keys()))
            key_l = v_dict.keys()
            for i in range(0, len(v_dict.keys())-1):
                try:
                    if( v_dict[key_l[i]] == v_dict[key_l[i+2]] ):
                        v_dict.pop(key_l[i+2])
                        del key_l[i+2]
                        v_dict.pop(key_l[i+1])
                        del key_l[i+1]
                except KeyError or IndexError as e:
                    print ('delete all round point')
                    break
                
                if( v_dict[key_l[i]] == v_dict[key_l[i+1]] ):
                    v_dict.pop(key_l[i+1])
                    del key_l[i+1]

        else:
            print ("error,it's not a mutiple deformations" )
            return 0
            
    #get the average brightness
    #Brightness:sum of brightness of all pixel
    #num_p:number of pixel in this area
    print (Brightness, num_p)
    Ave_Brightness = 0 if num_p == 0 else Brightness / num_p

    return [Area, Ave_Brightness]
