#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from time import time
import sys
import json
import base64
from wangyx import getAreaBrightness


def clip(src,ploy):
    '''
    clip the src to dst with ploy as mask.
    '''
    t1 = time()
    empimg = np.zeros(src.shape, np.uint8)
    mask = empimg.copy()
    t2 = time()
    cv2.fillPoly(mask,ploy,(255,255,255))
    t3 = time()
    cnt = int(0)
    tmb = int(0)# temp bright
    b = float(0) # sum bright
    maxb = int(0)# max bright
    (h,w) = src.shape[0],src.shape[1]
    lx,ly,rx,ry = 0,0,w-1,h-1
    for e in ploy[0]:
        [x,y] = e
        lx = x if x<lx else lx
        rx = x if x>rx else rx
        ly = y if y<ly else ly
        ry = y if y>ry else ry
    
    maxp =0,0,(lx,ly)
    #==============================================
    # switch algorithm (交换算法)
    #==============================================
    test_wangyx = True
    if test_wangyx:
        cnt, avgb = getAreaBrightness(ploy, src)
        b = cnt*avgb
    elif len(src.shape) ==2:# gray image
        for i in range(ly,ry):
            for j in range(lx,rx):
                if mask[i][j] == 0:
                    continue
                tmb = src[i][j]
                mask[i][j] = tmb
                cnt +=1
                b += tmb
                maxb = tmb if tmb > maxb else maxb
                if tmb == maxb:
                    maxp = [j,i]
    else: #color image
        for i in range(src.shape[0]-1):
            for j in range(src.shape[1]-1):
                if mask[i][j][0] == 0:
                    continue
                tmb = max(max(src[i][j][0],src[i][j][1]),src[i][j][2])
                mask[i][j] = src[i][j]
                cnt +=1
                b += tmb
                maxb = tmb if tmb > maxb else maxb
                if tmb == maxb:
                    maxp = [j,i]
    t4 = time()
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #print mask
    #dstimg = cv2.filter2D(src, -1, mask)
    return int(b/max(1,cnt)), maxb, maxp, cnt, int(b),1000*(t4-t1)
    #return '"avg_bright":{0},"max_bright":{1},"max_point":{2},
    # \r\n"area":{3},"bright_weaght":{4}, "time_span":{5}'.format(int(b/cnt), maxb, maxp, cnt, int(b),1000*(t4-t1))
    #print 1000*(t2-t1), 1000*(t3-t2), 1000*(t4-t3)
    #return mask
    
def getperson(a,b,c,d,A,B,C,D):
    '''
    (a,c),(b,d) is the front rectangle
    (A,C),(B,D) if the background rectangle
    '''
    return [[b,c],[b,d],[B,D],  [B,C],[A,C],[a,c]]
    #return [[a,c],[b,c], [b,d], [B,D], [B,C],[A,C]]

# 没有引用
def findperson(plts,bx0,bx1,bx2,bx3,b0,b1,b2,b3, t):
    '''
    xx is the front rectangle
    (A,C),(B,D) if the background rectangle
    '''
    return [[b,c], [b,d], [B,D], [B,C],[b,c],[a,c],[A,C],[B,C]]

#没有引用
def getPloy(xx,b1,b2,b3):
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
def getBed(img, x, yy, cl=(255,128,0)):
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

#没有引用
def lineBed(img, x, yy, cl=(255,128,0)):
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
    xx,b0,b1,b2,b3 = param
    img = cv2.imread(file_src)
    title = 'cv2py'
    cv2.namedWindow(title)
    #cv2.imshow(title, img)
    #cv2.waitKey(500)
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = gray
    #cv2.imshow(title, gray)
    #cv2.waitKey(500)
   
    retval,bina = cv2.threshold(gray, 120, 255, 0)
    #cv2.imshow(title, bina)
    #cv2.waitKey(500)
   
    imblur = cv2.GaussianBlur(bina, (3,3), 1.5)
    canny = cv2.Canny(imblur, 50, 130, 3)
    #cv2.imshow(title, canny)
    #cv2.waitKey(500)
   
    cts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
    for i in range(0, len(cts)):
        br = cv2.boundingRect(cts[i])    #计算轮廓的垂直边界最小矩形
        ar = cv2.contourArea(cts[i])
        ac = cv2.arcLength(cts[i], False)  #检测物体的周长
        mu = cv2.moments(cts[i], False)   #计算图像中的中心矩(最高到三阶)
        if len(cts[i]) > 200:
            #print ((br, ar, ac))
            #print (mu)
            #print ('\n')
            pass
    
    h,w = img.shape[0],img.shape[1]
    cen = (w//2,h//2)

     # xx,b0,b1,b2,b3 = param
    xx = [int(e*w) for e in xx]#[0,0.204166667,0.159027778,0.305555556,0.24375,0.355555556,0.644444444,0.75625,0.694444444,0.840972222,0.795833333,1]
    b0 = [int(e*h) for e in b0]#[0.0        ,0.257986111,0.387986111,0.664236111,0.938541667, 1.0,        1.0]
    b1 = [int(e*h) for e in b1]#[0.0        ,0.243055556,0.407986111,0.564236111,0.838541667, 1.0,        1.0]
    b2 = [int(e*h) for e in b2]#[0.118055556,0.329861111,0.421875,   0.522569444,0.71875,     0.777777778,0.9375]
    b3 = [int(e*h) for e in b3]#[0.239583333,0.375661376,0.439236111,0.5,        0.659722222, 0.708333333,0.777777778]
    
    #lineBed(img, [xx[0],xx[1],xx[10],xx[11]], b1, (255,128,0))
    #lineBed(img, [xx[2],xx[3],xx[8] ,xx[9]],  b2, (0,128,255))
    #lineBed(img, [xx[4],xx[5],xx[6] ,xx[7]],  b3, (128,225,128))
    '''
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
    '''
    x=xx
    bx0 = [0,0, w-1,w-1]
    bx1 = [x[0], x[1], x[11], x[10]]
    bx2 = [x[2], x[3], x[9], x[8]]
    bx3 = [x[4], x[5], x[7], x[6]]
    lie_plts, sit_plts, std_plts=[],[],[]
    lie_plts.append(getperson(bx0[0],bx0[1],b0[2],b0[3],bx1[0],bx1[1],b1[2],b1[3]))
    lie_plts.append(getperson(bx0[2],bx0[3],b0[2],b0[3],bx1[2],bx1[3],b1[2],b1[3]))
    lie_plts.append(getperson(bx1[0],bx1[1],b1[2],b1[3],bx2[0],bx2[1],b2[2],b2[3]))
    lie_plts.append(getperson(bx1[2],bx1[3],b1[2],b1[3],bx2[2],bx2[3],b2[2],b2[3]))
    lie_plts.append(getperson(bx2[0],bx2[1],b2[2],b2[3],bx3[0],bx3[1],b3[2],b3[3]))
    lie_plts.append(getperson(bx2[2],bx2[3],b2[2],b2[3],bx3[2],bx3[3],b3[2],b3[3]))
    lie_plts.append(getperson(bx0[0],bx0[1],b0[4],b0[5],bx1[0],bx1[1],b1[4],b1[5]))
    lie_plts.append(getperson(bx0[2],bx0[3],b0[4],b0[5],bx1[2],bx1[3],b1[4],b1[5]))
    lie_plts.append(getperson(bx1[0],bx1[1],b1[4],b1[5],bx2[0],bx2[1],b2[4],b2[5]))
    lie_plts.append(getperson(bx1[2],bx1[3],b1[4],b1[5],bx2[2],bx2[3],b2[4],b2[5]))
    lie_plts.append(getperson(bx2[0],bx2[1],b2[4],b2[5],bx3[0],bx3[1],b3[4],b3[5]))
    lie_plts.append(getperson(bx2[2],bx2[3],b2[4],b2[5],bx3[2],bx3[3],b3[4],b3[5]))

    sit_plts.append(getperson(bx0[0],bx0[1],b0[1],b0[2],bx1[0],bx1[1],b1[1],b1[2]))
    sit_plts.append(getperson(bx0[2],bx0[3],b0[1],b0[2],bx1[2],bx1[3],b1[1],b1[2]))
    sit_plts.append(getperson(bx1[0],bx1[1],b1[1],b1[2],bx2[0],bx2[1],b2[1],b2[2]))
    sit_plts.append(getperson(bx1[2],bx1[3],b1[1],b1[2],bx2[2],bx2[3],b2[1],b2[2]))
    sit_plts.append(getperson(bx2[0],bx2[1],b2[1],b2[2],bx3[0],bx3[1],b3[1],b3[2]))
    sit_plts.append(getperson(bx2[2],bx2[3],b2[1],b2[2],bx3[2],bx3[3],b3[1],b3[2]))
    sit_plts.append(getperson(bx0[0],bx0[1],b0[3],b0[4],bx1[0],bx1[1],b1[3],b1[4]))
    sit_plts.append(getperson(bx0[2],bx0[3],b0[3],b0[4],bx1[2],bx1[3],b1[3],b1[4]))
    sit_plts.append(getperson(bx1[0],bx1[1],b1[3],b1[4],bx2[0],bx2[1],b2[3],b2[4]))
    sit_plts.append(getperson(bx1[2],bx1[3],b1[3],b1[4],bx2[2],bx2[3],b2[3],b2[4]))
    sit_plts.append(getperson(bx2[0],bx2[1],b2[3],b2[4],bx3[0],bx3[1],b3[3],b3[4]))
    sit_plts.append(getperson(bx2[2],bx2[3],b2[3],b2[4],bx3[2],bx3[3],b3[3],b3[4]))

    std_plts.append(getperson(bx0[0],bx0[1],b0[0],b0[1],bx1[0],bx1[1],b1[0],b1[1]))
    std_plts.append(getperson(bx0[2],bx0[3],b0[0],b0[1],bx1[2],bx1[3],b1[0],b1[1]))
    std_plts.append(getperson(bx1[0],bx1[1],b1[0],b1[1],bx2[0],bx2[1],b2[0],b2[1]))
    std_plts.append(getperson(bx1[2],bx1[3],b1[0],b1[1],bx2[2],bx2[3],b2[0],b2[1]))
    std_plts.append(getperson(bx2[0],bx2[1],b2[0],b2[1],bx3[0],bx3[1],b3[0],b3[1]))
    std_plts.append(getperson(bx2[2],bx2[3],b2[0],b2[1],bx3[2],bx3[3],b3[0],b3[1]))

    cv2.imshow(title, img)
    cv2.waitKey(30)
    output = []
    for e in lie_plts:
        #print (e)
        #画多边形
        cv2.polylines(img,[np.array(e)],True,(np.random.randint(255), np.random.randint(255),np.random.randint(255)))
        cv2.imshow('test',img)
        cv2.waitKey(50)
        output.append( (e,'lie',clip(img, [np.array(e)])))
    cv2.imshow(title, img)
    cv2.waitKey(50)
    for e in sit_plts:
        #print (e)
        cv2.polylines(img,[np.array(e)],True,(np.random.randint(255), np.random.randint(255),np.random.randint(255)))
        output.append( (e,'sit',clip(img, [np.array(e)])))
        #cv2.fillPoly(img,[np.array(e)],(np.random.randint(255), np.random.randint(255),np.random.randint(255)))
    cv2.imshow(title, img)
    cv2.waitKey(50)
    for e in std_plts:
        #print (e)
        cv2.polylines(img,[np.array(e)],True,(np.random.randint(255), np.random.randint(255),np.random.randint(255)))
        output.append( (e,'std',clip(img, [np.array(e)])))
        #dstimg = clip(img, [np.array(e)])
    for e in output:
        #pplt, nm, (avb, maxb, maxp, area, bw, tmsp) = e
        #print '"avg_bright":{0},"max_bright":{1},"max_point":{2},\n"area":{3},"bright_weaght":{4}, "time_span":{5}\n'.format(avb, maxb, maxp, area, bw, tmsp)
        pass
    cv2.imshow(title, img)
    ret = cv2.waitKey(30)
    cv2.destroyAllWindows()
    #print '####################', ret
    return img,output

if __name__ == "__main__":
    '''
    usage:   c2py.py [src_image_file [threshold [output_file [parameter_file] ]]]

    input:
    file_src: the image file which is to be processed.
    output_file: the output .txt file
    threshold: the parameter to diceside whether pepole are.
    param:    the parameters to cut the image   (parameters参数)
    '''
    #p = urllib.urlopen(url)
    #image = np.asarray(bytearray(p.read()), dtype="uint8")
    #image = cv2.imdecode(image, cv2.GRAYSCALE)
    
    file_src = '3.png'
    dstfile = 'output.txt'
    #file_src = 'd:\\temp\\5.png'
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
    #命令行输入参数
    if len(sys.argv)>1:
        file_src = sys.argv[1]
    if len(sys.argv)>2:
        dstfile = sys.argv[2]
    if len(sys.argv)>3:
        threshold = int(sys.argv[3])
    if len(sys.argv)>4:
        file_param = sys.argv[4]
        with open(file_param, 'r') as f:
            param = json.load(f)
    file_bin = sys.argv[0]
        
    # begin main process
    img,output = main_proc(file_src, param, threshold)


    dic_out = {"img":"","beds":[{},{},{},{},{},{},{},{},{},{},{},{}]}
    f1 = sys.argv[0]
    # pars
    for i,e in enumerate(output):#将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        pplt, nm, (avb, maxb, maxp, area, bw, tmsp) = e
        info1 =  '_______body_is_'+nm
        if avb<threshold:
            info1 = 'no_body'
        info = bed[i]+info1
        ostr ='"name":"{6}","position":"{7}",\n"avg_bright":{0},\n"max_bright":{1},\n"max_point":{2},' \
              '\n"area":{3},\n"bright_weaght":{4}, \n"time_span":{5}\n'.format(avb, maxb, maxp, area, bw, tmsp, info,pplt )
        if i<12:
            person[i] = ostr
            dic_out['beds'][i%12]['NO'] = str(i+1)
            dic_out['beds'][i%12]['status'] = 0 if avb < threshold else 1
        elif avb > threshold:
            person[i%12] = ostr
            dic_out['beds'][i%12]['status'] = 2
        
    cv2.imwrite("im1.png",img)
    b64 = base64.b64encode(open("im1.png","rb").read()).decode('utf-8')
    dic_out['img']=b64

    json_str = json.dumps(dic_out)
    # write to output file.
    of = open(dstfile, 'w')
    if None != of:
        try:
            of.write(json_str)
            for e in person:
                print (e)
                of.write(''.join(e) )
            of.close()
        finally:
            pass
    
    #exit(0)
'''
    oimg=img

    cy,cx,cxoff,cyoff, xoff = h//2, w//2,w*(200-35)/2//720, h*135/2//576, w*215/720//2
    yoffc, yoffb = h*120//720, cy
    cyl1,cyl2,cyl3,cyl0 = h*120//720, h*90//720, h*45//720, h*30//720
    cyoffd,cyoffc,cyoffu = cy+yoffc,cy,cy-yoffc
    h1,ch,mh = h//10, h//2,h-1
    w1,cw,mw = w//10, w//2,w-1
    #0.76~0.87
    cxoffi = w*(200-35+70)/2//720
    pru,prd,plu,pld = (cw+cxoff,ch),(cw+cxoff,ch+cyl1),(cw-cxoff,ch),(cw-cxoff,ch+cyl1)
    iru,ird,ilu,ild = (cw+cxoffi,ch-cyl0),(cw+cxoffi,ch+cyl2),(cw-cxoffi,ch-cyl0),(cw-cxoffi,ch+cyl2)
    rui1,rui2,rui3,rui4 = (cw+cxoffi,ch-cyl3),(cw+cxoffi,ch-cyl2),(cw+cxoffi,ch-cyl1-cyl0),(cw+cxoffi,ch-cyl1-cyl0)
    lui1,lui2,lui3,lui4 = (cw-cxoffi,ch-cyl3),(cw-cxoffi,ch-cyl2),(cw-cxoffi,ch-cyl1-cyl0),(cw-cxoffi,ch-cyl1-cyl0)

    p1,p2,p3,p4 = (int(w*0.87),mh),(mw,mh),(w,int(h*0.80)),(w,int(h*0.6))
    p5,p6,p7,p8 = (w,int(h*0.415)), (w,int(h*0.115)), (w,0),(int(w*0.87),0)

    color, co2, co3=(192,128,0), (255,0,255), (64,128,255) #blue pink, oragn
    cv2.line(oimg, prd, p1, color)
    #cv2.line(oimg, prd, p2, color)
    cv2.line(oimg, ird, p3, color)
    cv2.line(oimg, pru, p4, color)
    cv2.line(oimg, rui1, p5, color)
    cv2.line(oimg, rui2, p6, co2)
    #cv2.line(oimg, rui3, p7, color)
    cv2.line(oimg, rui4, p8, color)

    lp1,lp2,lp3,lp4 = (int(w*0.13),mh),(0,mh),(0,int(h*0.80)),(0,int(h*0.6))
    lp5,lp6,lp7,lp8 = (0,int(h*0.415)), (0,int(h*0.115)), (0,0),(int(w*0.13),0)


    # radio line
    cv2.line(oimg, pld, lp1, color)
    #cv2.line(oimg, pld, lp2, color)
    cv2.line(oimg, ild, lp3, color)
    cv2.line(oimg, plu, lp4, color)
    cv2.line(oimg, lui1, lp5, color)
    cv2.line(oimg, lui2, lp6, co2)
    #cv2.line(oimg, lui3, lp7, color)
    cv2.line(oimg, lui4, lp8, color)

    #row line
    row1a,row1b=((0,h*495//576),(mw, h*495//576))
    row2a,row2b=((0,h*440//576),(mw, h*440//576))
    row3a,row3b=((0,h*415//576),(mw, h*415//576))
    row4a,row4b=((0,h*392//576),(mw, h*392//576))
    row5a,row5b=((0,h*380//576),(mw, h*380//576))
    cv2.line(oimg, row1a,row1b,co2)
    cv2.line(oimg, row2a,row2b,co2)
    cv2.line(oimg, row3a,row3b,co2)
    cv2.line(oimg, row4a,row4b,co2)
    cv2.line(oimg, row5a,row5b,co2)

    #col line
    col1a,col1b=((w*133//720, 0),(w*133//720, mh))
    col2a,col2b=((w*195//720, 0),(w*195//720, mh))
    col3a,col3b=((w*240//720, 0),(w*240//720, mh))
    col4a,col4b=((w*274//720, 0),(w*274//720, mh))
    col5a,col5b=((w*360//720, 0),(w*360//720, mh))
    col6a,col6b=((w*480//720, 0),(w*480//720, mh))
    col7a,col7b=((w*519//720, 0),(w*519//720, mh))
    col8a,col8b=((w*564//720, 0),(w*564//720, mh))
    col9a,col9b=((w*626//720, 0),(w*626//720, mh))
    cv2.line(oimg, col1a,col1b,co3)
    cv2.line(oimg, col2a,col2b,co3)
    cv2.line(oimg, col3a,col3b,co3)
    cv2.line(oimg, col4a,col4b,co3)
    cv2.line(oimg, col5a,col5b,co3)
    cv2.line(oimg, col6a,col6b,co3)
    cv2.line(oimg, col7a,col7b,co3)
    cv2.line(oimg, col8a,col8b,co3)
    cv2.line(oimg, col9a,col9b,co3)
    
    lxb = [(720-493)/1440.0, (720-277-77)/1440.0, (720+277+77)/1440.0, (720+493)/1440.0]
    lxi = [(720-396)/1440.0, (720-200)/1440.0,    (720+200)/1440.0,    (720+396)/1440.0]
    lyb = [0.0,0.115,0.415,0.6,0.8,1.0]
    lyi = [138/576.0,213/567.0,243/576.0,0.5,348/576.0,408/576.0]
    rh  = [0.6597,0.6805,0.7205,0.7639,0.8594]
    xb  = [int(e*w) for e in lxb]
    xi  = [int(e*w) for e in lxi]
    yr  = [int(e*h) for e in lyi]

    print xb,xi,yr
    #plt.plot(xi[0],yr[0],'ro',label='point')
    #plt.legend()
    #plt.show()
'''
