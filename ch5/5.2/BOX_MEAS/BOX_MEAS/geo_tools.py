#!/usr/bin/python3
# coding=utf-8

####################
# 几何运算工具
####################
# 20180605
#-------------------
# 修正调用np.linalg.pinv时
# 类型匹配问题(入参改成np.float64)
####################

import math
import numpy as np
import cv2

EPS=1e-16           # 防止除0错的分母修正因子
USE_CV_FUNC=True    # 尽量使用openCV的内部函数实现

## 功能描述：
#   2D坐标旋转
# 输入
#   x,y：坐标列表(向量）
#   sin_a,cos_a：旋转角度的正弦和余弦（2个标量）
# 输出
#   x_rot,y_roy：旋转后的坐标列表（向量）
def coord_rot_raw_2d(x,y,cos_a,sin_a):
    x_rot=cos_a*x-sin_a*y
    y_rot=cos_a*y+sin_a*x
    return x_rot,y_rot


## 功能描述：
#   2D坐标旋转
# 输入
#   x,y：坐标列表(向量）
#   a：旋转角度（标量）
# 输出
#   x_rot,y_roy：旋转后的坐标列表（向量）
def coord_rot_2d(x,y,a): return coord_rot_raw_2d(x,y,np.cos(a),np.sin(a))


## 功能描述：
#   找到1D数组中的第一个非零数据
# 输入
#   data：1维数组
# 输出
#   第一个非零元素序号，如果是-1，表示没有非零元素
def find_1st_nonzero(data):
    idx=np.nonzero(data)[0]
    return idx[0] if len(idx)>0 else -1

    
## 功能描述：
#   2D深度图中射线长度探测，基于深度变化门限，找出最长射线，返回长度和射线端点
# 输入：
#   img_dep: 深度图（2D）
#   x,y：    射线起点
#   a:      射线角度（顺时钟方向）
#   length: 最大探测距离（像素长度）
#   th:     射线断点判断门限（深度变化量门限）
#   step:   探测步长（像素为单位）
# 返回：
#   1. 长度
#   2. 射线断点x/y坐标
def half_line_det(img_dep,a,x,y,length,th,step=2):
    # 探测的直线坐标序列
    xline=np.round(np.arange(0,length,step).astype(float)*np.cos(a)+float(x)).astype(int)
    yline=np.round(np.arange(0,length,step).astype(float)*np.sin(a)+float(y)).astype(int)
        
    # 去除屏幕外的探测点
    hgt,wid=img_dep.shape
    valid=np.bitwise_and(np.bitwise_and(xline<wid,xline>=0),
                         np.bitwise_and(yline<hgt,yline>=0))
    if np.sum(valid)==0: return 0,x,y
    xdet=xline[valid]
    ydet=yline[valid]
    
    # 沿探测直线的深度采样，计算深度改变量
    dep=img_dep[ydet,xdet]
    dep_diff=np.abs(dep[1:]-dep[:len(dep)-1])
    
    # 沿射线找到第一个深度突变点, 返回深度突变点位置以及坐标
    idx=find_1st_nonzero(dep_diff>th)       # （如果没有突变点，idx=-1）
    length=len(xdet)*step if idx<0 else idx*step   # 射线长度
    
    return length,xdet[idx],ydet[idx]


## 功能描述：
#   2D深度图中条状区域宽度测量：基于射线长度探测和深度变化门限检测
#   如果作用于不规则区域检测，返回搜索角度范围内，最“窄”方向的角度和对应宽度
# 输入：
#   img_dep:2D深度图
#   x,y:    射线起点
#   a_range:探测角度（顺时钟方向，由于探测会沿正反两个方向进行，探测角度范围不必超过180度）
#   length: 探测距离（像素长度）
#   th:     射线断点判断门限（深度变化量门限）
#   step:   探测步长（像素为单位）
# 返回：
#   1. 宽度
#   2. 角度
# TODO:
#   图像预先计算为梯度图，可以批量测量
def bar_width_det(img_dep,a_range,x,y,length,th,step=1):    
    w=np.inf    # 宽度初始值
    aw=0        # 宽度测量的角度
    for a in a_range:
        len0,_,_=half_line_det(img_dep,a,x,y,length,th,step)        # 射线正方向长度探测
        len1,_,_=half_line_det(img_dep,np.pi+a,x,y,length,th,step)  # 射线反方向长度探测
        if len0+len0<w:
            w=len0+len1 # 沿正反两个方向的射线长度和
            aw=a        # 探测角度
    return w,aw


## 功能描述：
#   条状区域宽度快速比较,检测是否比2r细
#   检测距离(x,y)为r，角度范围a_range的弧线是否在(x,y)所在的条状区域外（通过比较弧线和(x,y)处深度差别实现）
# 输入：
#   img_dep:深度图
#   a_range:探测角度范围（由于探测会沿正反两个方向进行，探测角度范围不必超过180度）
#   x,y:    探测区域中心点
#   r:      探测半径
#   th:     深度门限
# 返回：
#   True/False: 指示条状区域宽度是否小于2r
def bar_width_compare(img_dep,a_range,x,y,r,th):
    # 计算a_range对应的弧线坐标(xarc,yarc)
    dx=np.round(np.cos(a_range)*r).astype(int)
    dy=np.round(np.sin(a_range)*r).astype(int)
    xarc=x+np.concatenate((dx,-dx)) # 坐标计算考虑了圆弧相对的两个方向
    yarc=y+np.concatenate((dy,-dy))
    
    # 屏蔽图像区域外的弧线坐标序列
    hgt,wid=img_dep.shape
    valid=np.bitwise_and(np.bitwise_and(xarc>=0,xarc<wid),
                         np.bitwise_and(yarc>=0,yarc<hgt))
    if np.sum(valid)==0: return False   # 没有有效坐标
    xarc=xarc[valid]
    yarc=yarc[valid]
    
    # 弧线深度改变量
    dep_diff=np.abs(img_dep[yarc,xarc]-img_dep[y,x])
    
    return find_1st_nonzero(dep_diff>th)>=0


## 功能描述：
#   2D深度图中条带区域检测
#   使用探测环方法，即以每个待分析的像素点为圆心，检测围绕它的圆环上点的深度，对比圆心深度找出“内点”（深度和圆心接近）和“外点”（背景，深度大于圆心）
#   当探测环的圆心处于条带中心，并且探测环跨越条带时，内点和外点对称分布，通过计算对称性得到“条带指数”，该指数越大表明探测环圆心越可能是条带中心
# 输入
#   img_dep:    2D深度图
#   mask:       深度图的有效区域屏蔽码（2D数组，元素True标识该像素有效，并需要被探测）
#   det_wid:    探测环的直径，必须大于线条宽度
#   th:         判定条带边界的深度值增量门限（认为条带外深度值突增）
#   downsample: 图像降采样率（大于1时，对原图降采样，用于提高运算速度）
#   angle_div:  探测“环”的角度分辨率（越大，角度探测越细，时间越长）
#   point_th_hi:探测“环”上内点（近距离点）比例上限。这是为了确保探测宽度大于线宽，这样探测中心对应的探测点能够跨越直线两侧
#   point_th_lo:探测“环”上内点（近距离点）比例下限。这是为了确保探测宽度不要过大，这样探测中心对应的探测点中有足够多的内点
#   eps:        计算条带区域可能性指数时，防止除0错的补偿因子
# 输出
#   2D标记图，元素值对应条带的可能性，越大越可能是条带区域
def bar_det(img_dep,mask,det_wid,f,th=100.0e-3,downsample=3,angle_div=24,point_th_hi=0.9, point_th_lo=0.1,det_eps=0.1):
    img_dep=img_dep.copy()
    img_dep[~mask]=np.inf
    
    hgt,wid=img_dep.shape
    
    # 建立探测模板
    a=np.arange(angle_div)*(360.0/angle_div)*(np.pi/180.0)  # 探测圆弧采样点（探测角列表）
    cos_a,sin_a=np.cos(a),np.sin(a)
    xtmp,ytmp=det_wid/2.0*cos_a,det_wid/2.0*sin_a           # 探测点模板偏移量
    
    # 选出有效坐标列表，并提取其深度值
    if downsample>1:    # 探测点降采样
        mask_filter=np.zeros((hgt,wid),dtype=bool)
        mask_filter[::downsample,::downsample]=True
        mask=np.bitwise_and(mask,mask_filter)
    ysel,xsel=np.where(mask)    # 坐标列表  
    dsel=img_dep[ysel,xsel]     # 坐标列表对应的深度值
    
    dscale=1.0/(dsel+EPS)*f # 物理尺度缩放因子，将物理偏移量折算成像素偏移量

    # 对应每个探测中心，计算探测点坐标
    xoff=np.kron(dscale,xtmp).reshape(len(xsel),len(xtmp))    # 构成矩阵，每行对应一个探测中心点的各个探测点坐标
    yoff=np.kron(dscale,ytmp).reshape(len(ysel),len(ytmp))
    
    xcent,ycent=xsel,ysel               # 探测中心列表（向量）
    xcent_ext=np.kron(xcent,np.ones(len(xtmp))).reshape(len(xcent),len(xtmp))
    ycent_ext=np.kron(ycent,np.ones(len(ytmp))).reshape(len(ycent),len(ytmp))
    xdet=(xcent_ext+xoff).astype(int)   # 对应每个探测中心的探测点矩阵（每行对应一个探测中心的每个探测点坐标）
    ydet=(ycent_ext+yoff).astype(int)

    # 滤除屏幕外的探测点
    valid=~np.bitwise_or(np.bitwise_or(np.any(xdet<0,axis=1),np.any(xdet>=wid,axis=1)),
                         np.bitwise_or(np.any(ydet<0,axis=1),np.any(ydet>=hgt,axis=1))) # 有效点标记
    xdet,ydet=xdet[valid],ydet[valid]       # 探测点
    xcent,ycent=xcent[valid],ycent[valid]   # 探测中心
    if len(xcent)==0:
        return np.zeros((hgt,wid),dtype=float),xcent,ycent,None
    
    # 深度的探测
    ddet=img_dep[ydet,xdet].astype(float)   # 每行对应各个探测中心的探测点深度
    dcent=img_dep[ycent,xcent]              # 探测中心深度
    dcent_ext=np.kron(dcent,np.ones(len(xtmp))).reshape(len(dcent),len(xtmp)).astype(float)
    dcomp=ddet<dcent_ext+th                 # 标识内点（探测点深度小于中心点，标明是内点）
    
    # 过滤掉内点过多或者过少的情况，即：某个中心点附近内点太多或者太少
    sel_0=np.sum(dcomp,axis=1)<(float(len(cos_a))*point_th_hi)# 内点不能太多，确保探测宽度大于线宽，这样探测中心对应的探测点能够跨越直线两侧
    sel_1=np.sum(dcomp,axis=1)>(float(len(cos_a))*point_th_lo)# 内点不能太少，确保探测宽度不要过大，这样探测中心对应的探测点中有足够多的内点
    sel=np.bitwise_and(sel_0,sel_1)
    xcent,ycent=xcent[sel],ycent[sel]
    dcomp=dcomp[sel,:]

    # 通过探测环上的内点方向性（方向向量之和）
    # 探测中心如果在条带中心，则方向性越低（由于内点和外点的在探测环上的位置对称性）
    tmp=(np.tile(cos_a,np.size(dcomp,axis=0))*dcomp.flatten().astype(int))
    ddir_cos=np.mean(tmp.reshape(dcomp.shape),axis=1)
    
    tmp=(np.tile(sin_a,np.size(dcomp,axis=0))*dcomp.flatten().astype(int))
    ddir_sin=np.mean(tmp.reshape(dcomp.shape),axis=1)
    
    ddir=np.sqrt(ddir_cos**2+ddir_sin**2)   # 取向程度探测，越大表示越有取向性
    
    # 条带指数计算，越大表明越可能是带探测条带中心
    bar_idx=1/(ddir+det_eps)
    
    # 以2D图形式标出各点的条带指数
    img_bar_idx=np.zeros((hgt,wid),dtype=float)    
    img_bar_idx[ycent,xcent]=bar_idx
    
    return img_bar_idx,xcent,ycent,bar_idx


## 功能描述：
#   2D直线拟合，计算拟合模型: y=a+bx 对应的(a,b)参数
#   使用最小二乘拟合
# 输入：
#   x,y: 待拟合直线坐标列表（向量）
# 输出：
#   a,b: 直线拟合的模型参数
def line_fit_2d(x,y):
    num=len(x)
    if num==0: return 0,0
    
    xbar=np.mean(x)
    ybar=np.mean(y)
    
    tmp=np.sum(x**2)-num*xbar*xbar
    if np.abs(tmp)<EPS:
        tmp=EPS if tmp>0 else -1.0*EPS
    
    b=(np.sum(x*y)-num*xbar*ybar)/tmp
    a=ybar-b*xbar
    return a,b


## 功能描述：
#   计算2D平面内，点到直线的距离
# 输入：
#   a,b： 直线模型参数，y=a+bx
#   x,y： 待计算的2D点坐标列表（向量）
# 返回：
#   d： 距离列表（向量）
def line_distance_2d(a,b,x,y): return (1.0/np.sqrt(1+b*b))*(a+b*x-y)


## 功能描述：
#   3D空间点拟合直线参数
#   拟合模型为：(x-x0)/a=(y-y0)/b=(z-z0)/c
#   使用TLS算法
# 输入：
#   x,y,z: 待拟合直线坐标列表（向量）
# 输出：
#   a,b,c,x0,y0,z0: 直线拟合的模型参数
#   cond: 直线的取向程度（越大，取向越明显，是点云协方差阵的最长次长轴之比）
def line_fit_3d(x,y,z):
    if len(x)==0: 
        #print('[WRN] line_fit_3d(), length of input coord is 0')
        return 0,0,0,0,0,0,0
    
    # 均值计算
    x0=np.mean(x)   
    y0=np.mean(y)
    z0=np.mean(z)
    
    # 计算协方差矩阵R
    x1=x-x0
    y1=y-y0
    z1=z-z0
    
    xx=np.sum(x1*x1)
    xy=np.sum(x1*y1)
    xz=np.sum(x1*z1)
    yy=np.sum(y1*y1)
    yz=np.sum(y1*z1)
    zz=np.sum(z1*z1)
    
    R=np.array([[xx,xy,xz],
                [xy,yy,yz],
                [xz,yz,zz]])
    
    # 方向向量计算，对应协方差矩阵最大特征值的特征向量
    e,V=np.linalg.eigh(R)
    a,b,c=V[0,2],V[1,2],V[2,2]
    
    cond=np.max(e)/(np.sum(e)-np.max(e)-np.min(e))
    #cond=e[2]/(e[1]+EPS)
    
    return a,b,c,x0,y0,z0,cond


## 功能描述：
#   3D空间点拟合直线参数，基于openCV
def line_fit_3d_cv(x,y,z):
    points=np.concatenate((x,y,z)).reshape(3,len(x)).T
    a,b,c, x0,y0,z0 = cv2.fitLine(points=points, 
                                  distType=cv2.DIST_L2, 
                                  param=0, 
                                  reps=0.01, 
                                  aeps=0.01)
    return a[0],b[0],c[0],x0[0],y0[0],z0[0]
    

## 功能描述：
#   计算3D空间点到直线的距离
# 输入：
#   a,b,c,x0,y0,z0: 3D直线参数模型，(x-x0)/a=(y-y0)/b=(z-z0)/c
#   x,y,z: 待计算点坐标序列（向量）
# 输出：
#   每个点对应的距离
def line_distance_3d(a,b,c,x0,y0,z0,x,y,z):    
    sz=len(z)
    p1=np.array([x.ravel()-x0,y.ravel()-y0,z.ravel()-z0]).reshape(3,sz).T
    p2=np.array([x.ravel()-(a+x0),y.ravel()-(b+y0),z.ravel()-(c+z0)]).reshape(3,sz).T   
    return np.linalg.norm(np.cross(p1,p2),axis=1)/np.linalg.norm([a,b,c])
    

## 功能描述：
#   RANSAC方法计算2D空间点的直线拟合结果
# 输入：
#   xlist,ylist:    2D空间点的坐标列表（向量）
#   prob:           每次拟合直线抽取的点的数量比例（注意: 要求prob*总点数>=1)
#   dth:            距离门限，用于判定某点和是否归属当前拟合的直线
#   N:              重启随机抽样的次数
#   M:              每一轮直线参数拟合的迭代次数
# 输出：
#   a_opt,b_opt：   直线拟合参数，y=a_opt+b_opt*x 
#   sel_opt:        归属被拟合直线的点的标志，bool型（和xlist,ylist向量长度相同）
# TODO:
#   需要解决错误情况，比如：输入点数过少，int(sz*prob)==0，拟合点数量=2等
#   加速迭代，当拟合参数不再改变时，提前退出循环
def ransac_line_2d(xlist,ylist,prob=0.2,dth=5,N=10,M=4):
    sz=len(xlist)
    
    a_opt,b_opt=0,0
    sel_opt=None
    num_opt=-np.inf
    
    sz_prob=int(sz*prob)

    for _ in range(N):
        # 随机抽样
        sel=np.random.choice(np.arange(sz),sz_prob,replace=False)
        
        # 直线拟合迭代
        for _ in range(M):  
            xsel,ysel=xlist[sel],ylist[sel]     # 抽取待拟合点坐标
            if len(xsel)==0: continue
            a,b=line_fit_2d(xsel,ysel)          # 拟合直线:y=a+bx
            d=line_distance_2d(a,b,xlist,ylist) # 计算所有点到直线的距离
            sel=np.abs(d)<dth   # 更新用于直线拟合的点，选取近距离的点进入下一轮迭代
        
        # 记录最优拟合结果
        num=np.sum(sel)
        if num>num_opt:
            num_opt=num         # 靠近拟合直线的点的数量
            sel_opt=sel.copy()  # 靠近拟合直线的点
            a_opt,b_opt=a,b     # 直线参数
            
    return a_opt,b_opt,sel_opt


## 功能描述：
#   RANSAC方法进行3D点云中的直线拟合
# 输入：
#   xlist,ylist,zlist:  2D空间点的坐标列表（向量）
#   prob:               每次拟合直线抽取的点的数量比例（注意: 要求prob*总点数>=1)
#   dth:                距离门限，用于判定某点和是否归属当前拟合的直线
#   N:                  重启随机抽样的次数
#   M:                  每一轮直线参数拟合的迭代次数
# 输出：
#   a_opt,b_opt,c_opt,x0_opt,y0_opt,z0_opt：直线拟合参数，(x-x0_opt)/a_opt=(y-y0_opt)/b_opt=(z-z0_opt)/c_opt
#   sel_opt:            归属被拟合直线的点的标识，bool型（和xlist,ylist,zlist向量长度相同）
# TODO:
#   需要解决错误情况，比如：输入点数过少，int(sz*prob)==0，拟合点数量=2等
#   加速迭代，当拟合参数不再改变时，提前退出循环
def ransac_3d_line(xlist,ylist,zlist,prob=0.1,dth=0.2,N=20,M=10):
    sz=len(xlist)
    
    a_opt,b_opt,c_opt=0,0,0
    x0_opt=y0_opt=z0_opt=0,0,0
    sel_opt=None
    num_opt=-np.inf
    cd_opt=0
    
    for _ in range(N):
        # 随机抽样
        sel=np.random.choice(np.arange(sz),int(sz*prob),replace=False)
        
        # 直线拟合迭代
        for _ in range(M):  
            xsel,ysel,zsel=xlist[sel],ylist[sel],zlist[sel]     # 抽取待拟合点坐标
            if len(xsel)==0: continue
            if USE_CV_FUNC:
                a,b,c,x0,y0,z0   =line_fit_3d_cv(xsel,ysel,zsel)
            else:
                a,b,c,x0,y0,z0,cd=line_fit_3d(xsel,ysel,zsel)   # 拟合直线: (x-x0)/a=(y-y0)/b=(z-z0)/c
            
            d=line_distance_3d(a,b,c,x0,y0,z0,xlist,ylist,zlist)# 计算所有点到直线的距离
            sel=np.abs(d)<dth   # 更新用于直线拟合的点，选取近距离的点进入下一轮迭代

        # 记录最优拟合结果
        num=np.sum(sel)
        if num>num_opt:
            num_opt=num
            a_opt,b_opt,c_opt=a,b,c
            x0_opt,y0_opt,z0_opt=x0,y0,z0
            if not USE_CV_FUNC: cd_opt=cd   # 由于CV提供的拟合函数不能输出直线的取向性参数cd
            sel_opt=sel.copy()
            
        if USE_CV_FUNC:  # CV提供的拟合函数不能输出直线的取向性参数cd，因此需要单独计算
            _,_,_,_,_,_,cd_opt=line_fit_3d(xlist[sel_opt],ylist[sel_opt],zlist[sel_opt])
            
    return a_opt,b_opt,c_opt,x0_opt,y0_opt,z0_opt,sel_opt,cd_opt


## 功能描述：
#   根据像素坐标(u,v)和深度d计算空间坐标(x,y)
# 输入：
#   u,v     ：像素坐标
#   d       ：像素对应的深度
#   wid,hgt ：深度图的宽度和高度
#   f       ：f相机的内参数据
# 输出：
#   x,y     ：空间x,y坐标（z坐标就是入参d）
def calc_phy_coord(u,v,d,wid,hgt,f):
    x = (u-wid/2)*d/f
    y = (v-hgt/2)*d/f
    return x,y


## 功能描述：
#   计算物理距离
# 输入参数：
#   u1,v1,d1：第1个点的像素坐标和深度
#   u2,v2,d2：第2个点的像素坐标和深度
#   wid,hgt ：深度图的宽度和高度
#   f       ：f相机的内参数据
# 输出：
#   两点之间物理距离
def calc_phy_dist(u1,v1,d1,u2,v2,d2,wid,hgt,f):
    x1,y1=calc_phy_coord(u1,v1,d1,wid,hgt,f)
    x2,y2=calc_phy_coord(u2,v2,d2,wid,hgt,f)
    return np.sqrt((x1-x2)**2+(y1-y2)**2+(d1-d2)**2)


## 功能描述：
#   根据物理坐标(x,y,z)计算像素坐标(u,v)
# 输入参数：
#   wid,hgt ：深度图的宽度和高度
#   f       ：f相机的内参数据
# 输出：
#   u,v     ：像素坐标
def calc_pix_coord(x,y,z,wid,hgt,f):
    u=x/z*f+wid/2
    v=y/z*f+hgt/2   
    return int(round(u)),int(round(v))


## 功能描述：
#   深度图中的直线延伸检测
#   给出直线方程参数，检测直线的两头延伸，延伸中断以深度变量为准
# 输入：
#   vx,vy,vz,x0,y0,z0：直线模型(x-x0)/vx=(y-y0)/vy=(z-z0)/vz
#   img_dep：深度图
#   f：深度相机内参
#   length:最大探测（物理）距离
#   step: 探测步长（物理距离）
#   mask：探测的像素有效区域
# 输出：
#   xs_ext,ys_ext,zs_ext：延伸起点的物理坐标
#   us_ext,vs_ext       ：延伸起点的像素坐标
#   xe_ext,ye_ext,ze_ext：延伸终点的物理坐标
#   ue_ext,ve_ext       ：延伸终点的像素坐标
def line_extend(vx,vy,vz,x0,y0,z0,dth,img_dep,f,length=1.0,step=0.01,mask=None):
    # 深度图尺寸
    img_hgt,img_wid=img_dep.shape
    
    # 有效区域屏蔽码
    if mask is None: mask=np.ones((img_hgt,img_wid),dtype=bool)
        
    # 计算归一化探测方向矢量    
    tmp=np.linalg.norm([vx,vy,vz])
    dx,dy,dz=vx/tmp,vy/tmp,vz/tmp  # 单位长度向量（长度1M）
     
    # 探测的起止点
    xs,ys,zs=x0+dx*length,y0+dy*length,z0+dz*length
    xe,ye,ze=x0-dx*length,y0-dy*length,z0-dz*length
    
    u0,v0=calc_pix_coord(x0,y0,z0,img_wid,img_hgt,f)  # 手臂中心
    us,vs=calc_pix_coord(xs,ys,zs,img_wid,img_hgt,f)  # 手臂前向最远探测点
    ue,ve=calc_pix_coord(xe,ye,ze,img_wid,img_hgt,f)  # 手臂后向最远探测点
    
    # 每个方向总的探测步数
    total_step=int(round(length/step))
    
    # 前向探测，找到最远点(us_ext,vs_ext)
    d0=img_dep[v0,u0]
    us_ext,vs_ext=u0,v0 # 前向延伸终点     
    xs_ext,ys_ext,zs_ext=x0,y0,z0 
    for t in np.arange(total_step).astype(float)/float(total_step):
        u=int(round((us-u0)*t+u0))  # 计算前向延伸点
        v=int(round((vs-v0)*t+v0))
        if u<0 or v<0 or u>=img_wid or v>=img_hgt: 
            break   # 延伸超出屏幕，停止延伸
        if not mask[v,u]:
            break   # 延伸到图像无效区域，停止延伸
        d=img_dep[v,u]
        if abs(d-d0)>dth: 
            break   # 深度突变，延伸探索结束
        d0=d                # 更新深度参考值
        us_ext,vs_ext=u,v   # 延伸端点的像素坐标
        xs_ext=(xs-x0)*t+x0 # 延伸端点的空间坐标
        ys_ext=(ys-y0)*t+y0         
        zs_ext=(zs-z0)*t+z0         

    # 后向探测，找到最远点
    d0=img_dep[v0,u0]
    ue_ext,ve_ext=u0,v0 # 前向延伸终点            
    xe_ext,ye_ext,ze_ext=x0,y0,z0 
    for t in np.arange(total_step).astype(float)/float(total_step):
        u=int(round((ue-u0)*t+u0))  # 计算前向延伸点
        v=int(round((ve-v0)*t+v0))
        if u<0 or v<0 or u>=img_wid or v>=img_hgt: 
            break   # 延伸超出屏幕，停止延伸
        if not mask[v,u]:
            break   # 延伸到图像无效区域，停止延伸
        d=img_dep[v,u]
        if abs(d-d0)>dth: 
            break   # 深度突变，延伸探索结束
        d0=d                # 更新深度参考值
        ue_ext,ve_ext=u,v   # 延伸端点的像素坐标
        xe_ext=(xe-x0)*t+x0 # 延伸端点的空间坐标
        ye_ext=(ye-y0)*t+y0         
        ze_ext=(ze-z0)*t+z0         

    return xs_ext,ys_ext,zs_ext, us_ext,vs_ext, xe_ext,ye_ext,ze_ext, ue_ext,ve_ext
        

## 功能描述：
#   拟合球体
#   (x-x0)^2+(y-y0)^2+(z-z0)^2=R^2
# 输入参数：
#   pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#   shuffle_num: 拟合时数据顺序“换洗”次数
# 输出参数：
#   x0,y0,z0: 球心坐标
#   r: 球半径
def fit_sphere(pc,shuffle_num=1):
    num,_=pc.shape
    idx=np.arange(num)
    pc2=pc[:,0]**2+pc[:,1]**2+pc[:,2]**2
    
    for s in range(shuffle_num):
        np.random.shuffle(idx)
        D=pc-pc[idx,:]
        v=pc2-pc2[idx]
        if s==0:
            pinvR=np.linalg.pinv(np.dot(D.T,D).astype(np.float64))
            Dv=np.dot(D.T,v)
        else:
            pinvR=pinvR+np.linalg.pinv(np.dot(D.T,D).astype(np.float64))
            Dv=Dv+np.dot(D.T,v)
    
    # 球心坐标
    p0=np.dot(pinvR,Dv)*0.5
    x0,y0,z0=p0[0],p0[1],p0[2]
    
    # 球半径估计
    r=np.sqrt(np.mean((pc[:,0]-x0)**2+(pc[:,1]-y0)**2+(pc[:,2]-z0)**2))
    
    return x0,y0,z0,r


## 功能描述
#   pc点云拟合平面
#   nx*x+ny+y+nz+z+d=0
# 输入参数：
#   pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#   shuffle_num: 拟合时数据顺序“换洗”次数
# 输出参数：
#   nx,ny,nz：平面法向量
#   d：平面参数，满足nx*x+ny+y+nz+z+d=0
#   px,py,pz: 返回和平面经过的一个点坐标
def fit_plane_param(pc):
    num,_=pc.shape
    pc_ext=np.hstack((pc,np.ones((num,1))))
    R=np.dot(pc_ext.T,pc_ext)
    _,V=np.linalg.eigh(R.astype(np.float64))
    v0=V[:,0]
    
    # 法向量(长度归一化)
    nx,ny,nz,d=v0[0,],v0[1],v0[2],v0[3]
    k=1.0/np.sqrt(nx**2+ny**2+nz**2)
    nx*=k
    ny*=k
    nz*=k
    
    # 计算平面经过的一点
    d*=k
    px=py=pz=0
    if d!=0:
        if   nx != 0: px=-d/nx
        elif ny != 0: py=-d/ny
        elif nz != 0: pz=-d/nz
    return nx,ny,nz,d,px,py,pz
    

## 功能描述：
#   拟合空间圆环，找到环心(xc,yc,zc)和轴方向矢量(nx,ny,nz)
# 输入参数：
#   pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
# 输出：
#   xc,yc,zc：环心
#   nx,ny,nz：轴方向矢量
def fit_3D_circule(pc):
    # 找到环的轴线上的一个点(x0,y0,z0)
    x0,y0,z0,r=fit_sphere(pc)   
    
    # 移动圆环，使他的轴线经过原点
    pc0=pc.copy()
    pc0[:,0]-=x0
    pc0[:,1]-=y0
    pc0[:,2]-=z0
    
    # 计算圆环平面法向量
    nx,ny,nz,_,_,_,_=fit_plane_param(pc0)
    
    # 计算圆环上点在轴线上的投影并计算环心
    d=np.mean(pc0[:,0]*nx+pc0[:,1]*ny+pc0[:,2]*nz)
    xc,yc,zc=d*nx+x0,d*ny+y0,d*nz+z0
    
    # 计算环半径
    r=np.sqrt(np.mean((pc[:,0]-xc)**2+(pc[:,1]-yc)**2+(pc[:,2]-zc)**2))
    
    return xc,yc,zc,r,nx,ny,nz


## 度和弧度互换
def deg_to_rad(d): return d/180.0*math.pi
def rad_to_deg(d): return d*180.0/math.pi


## 计算两个向量夹角的cos值
def calc_xy_cos_angle(x1,y1,x2,y2): return (x1*x2+y1*y2)/math.sqrt((x1**2+y1**2)*(x2**2+y2**2))


## 计算向量(x,y)在向量(xr,yr)上的投影
def calc_xy_proj(x,y,xr,yr):
    k=(xr*x+yr*y)/(xr**2+yr**2)
    return k*xr,k*yr


## 计算向量(x,y)在向量(xr,yr)上的投影长度
def calc_xy_proj_len(x,y,xr,yr): return (xr*x+yr*y)/math.sqrt(xr**2+yr**2)


## 功能描述：
# 计算两个角度差，输出范围是-pi~pi
# 输入参数：
#   a1,a2两个角度
# 输出：
#   夹角，范围是-pi~pi
def calc_angle_diff(a1,a2): 
    d=(a1-a2)%(np.pi*2.0)
    return d if d<=np.pi else (d-np.pi*2.0)


## 功能描述：
# 计算两个角度的方向差别（考虑180反转可能）
# 输入参数：
#   a0,a1两个角度
# 输出：
#   夹角，范围是-pi~pi
def calc_bidir_angle_diff(a0,a1):
    d0=calc_angle_diff(a0,a1)
    d1=d0+np.pi
    if d1>np.pi: d1-=np.pi*2.0
    return d0 if abs(d0)<abs(d1) else d1 


## 功能描述：
#   计算向量v的2范数
def calc_vec_norm(v): return math.sqrt((v**2).sum())

## 功能描述：
#   计算两点间距平方
# 输入参数：
#   n0,n1   两点的xy坐标
def calc_point_dist_sqr(n0,n1): return (n0[0]-n1[0])**2+(n0[1]-n1[1])**2

## 功能描述：
#   计算两点间距平方
# 输入参数：
#   n0,n1   两点的xy坐标
def calc_point_dist(n0,n1): return math.sqrt(calc_point_dist_sqr(n0,n1))

## 功能描述：
#   计算直线段的倾角
# 输入参数：
#   n0,n1线段的两个端点的xy坐标
# 输出：
#   线段的倾角
def calc_line_angle(n0,n1): return math.atan2(n0[1]-n1[1],n0[0]-n1[0])


## 功能描述：
# 计算两个线段的夹角
# 输入参数：
#   n0,n1   第1条线段的两个端点的xy坐标
#   n2,n3   第2条线段的两个端点的xy坐标
# 输出：
#   线段的夹角
# 注意：
#   由于直线有两个夹角，函数只输出最接近0的角度
def calc_line_cross_angle(n0,n1,n2,n3):
    a0=calc_line_angle(n0,n1)
    a1=calc_line_angle(n2,n3)
    return calc_bidir_angle_diff(a0,a1)


## 功能描述：
#   计算两个线段的夹角余弦
# 输入参数：
#   n0,n1   第1条线段的两个端点的xy坐标
#   n2,n3   第2条线段的两个端点的xy坐标
# 输出：
#   夹角余弦
# 注意：
#   输出可能为负数
def calc_line_cross_cos_angle(n0,n1,n2,n3):
    na0,na1=n0[0]-n1[0],n0[1]-n1[1]
    nb0,nb1=n2[0]-n3[0],n2[1]-n3[1]
    return (na0*nb0+na1*nb1)/math.sqrt((na0**2+na1**2)*(nb0**2+nb1**2))
    
    
def in_angle_range(a,amin,amax):
    a=a%(np.pi*2.0)
    amin=amin%(np.pi*2.0)
    amax=amax%(np.pi*2.0)
    if amax<amin: amax,amin=amin,amax
    res=(a>=amin) and (a<=amax)
    return res if amax-amin<=np.pi else ~res


## 功能描述：
#   对于空间的p点，找到它直线上的投影
#   直线由端点n1和n2确定，
#   直线上的点为：n=n1*m+(1-m)*n2
#   求解的问题为：argmin ||p-n||^2
#   达到最小的m的解析解：
#   m = -<n1-n2,n2-p>/||n1-n2||^2
# 输入参数：
#   n1,n2线段的两个端点的xy坐标
# 输出：
#   投影pn及其对应的m
def calc_point_to_line_proj(n1,n2,p):
    m = -((n1-n2)*(n2-p)).sum()/(((n1-n2)**2).sum())
    pn=n1*m+(1-m)*n2
    return pn,m


## 功能描述：
#   计算p到(n1,n2)确定的直线的投影的距离（点p的投影可能在直线的延长线上）
# 输入参数：
#   n1,n2线段的两个端点的xy坐标
# 输出：
#   投影距离
def calc_point_to_line_dist_ext(n1,n2,p):
    pn,_=calc_point_to_line_proj(n1,n2,p)
    return np.sqrt(((pn-p)**2).sum())


## 功能描述：
#   计算p到线段(n1,n2)的距离
# 输入参数：
#   n1,n2线段的两个端点的xy坐标
# 输出：
#   最近距离
def calc_point_to_line_dist(n1,n2,p):
    pn,m=calc_point_to_line_proj(n1,n1,p)
    if m<0 or m>1:  # 投影在线段之外
        return min(calc_vec_norm(p-n1),calc_vec_norm(p-n2))
    else:           # 投影在线段中
        return calc_vec_norm(pn-p)


## 功能描述：
#   计算两个线段的最近距离
# 输入参数：
#   n1,n2   第1条线段的两个端点的xy坐标
#   n3,n4   第2条线段的两个端点的xy坐标
# 输出：
#   两种线段的最近距离
def calc_line_to_line_dist(n1,n2,n3,n4):
    d1=calc_point_to_line_dist(n3,n4,n1)
    d2=calc_point_to_line_dist(n3,n4,n2)
    
    d3=calc_point_to_line_dist(n1,n2,n3)
    d4=calc_point_to_line_dist(n1,n2,n4)
    
    return min(d1,d2,d3,d4)


## 功能描述：
#   计算点集的两两距离矩阵
# 输入参数：
#   point_set 存放待计算距离的点，每个元素数点的xy坐标对
# 输出：
#   距离矩阵
def calc_point_dist_mat(point_set):
    sz=len(point_set)
    dist_mat=np.zeros(sz,sz,dtype=np.float32)
    for m in range(sz):
        pm=point_set[m]
        for n in range(sz):
            pn=point_set[n]
            dist_mat[m,n]=np.sqrt(((pm-pn)**2).sum())
    return dist_mat


## 功能描述：
#   计算两条直线的交点
# 输入参数：
#   n1,n2   第1条线段的两个端点的xy坐标
#   n3,n4   第2条线段的两个端点的xy坐标
#   cond_max最大条件数
# 输出：
#   交点坐标，如果求解的矩阵接近奇异（直线段接近平行，由cond_max决定），则返回None
def calc_line_cross(n1,n2,n3,n4, cond_max=np.inf):
    M=np.array([[n1[0]-n2[0],n4[0]-n3[0]],
                [n1[1]-n2[1],n4[1]-n3[1]]])	# M=np.hstack(((n1-n2).reshape(2,1),(n4-n3).reshape(2,1)))
    v=np.array([[n4[0]-n2[0]],
	            [n4[1]-n2[1]]])				# v=(n4-n2).reshape(2,1)

    detM=M[0,0]*M[1,1]-M[0,1]*M[1,0]
	
	# if np.linalg.cond(M)>cond_max: return None
    if (M**2).sum()>cond_max*abs(detM): return None

    w=np.dot(np.linalg.inv(M),v)
    a,_=w.flatten()
    
    return np.array([(n1[0]-n2[0])*a+n2[0],(n1[1]-n2[1])*a+n2[1]])
    #p=(n1-n2)*a+n2
    #return p.flatten()

## 功能描述
#   计算四边形的面积
# 输入参数
#   nA,nB,nC,nD     4个顶点坐标, 注意，对角线是nA--nC和nB--nD
# 输出
#   面积
# 注意：确保nA、nB、nC和nD是顺时针或者逆时针排列
def calc_4point_area(nA,nB,nC,nD):
    a2=(nA[0]-nB[0])**2+(nA[1]-nB[1])**2
    b2=(nB[0]-nC[0])**2+(nB[1]-nC[1])**2
    c2=(nC[0]-nD[0])**2+(nC[1]-nD[1])**2
    d2=(nD[0]-nA[0])**2+(nD[1]-nA[1])**2

    D=abs(b2+d2-a2-c2)

    m2=(nA[0]-nC[0])**2+(nA[1]-nC[1])**2
    n2=(nB[0]-nD[0])**2+(nB[1]-nD[1])**2
    
    S=math.sqrt(max(4*m2*n2-D**2,0))*0.25
    return S


## 功能描述
#   生成直线采样点
# 输入参数
#   n0,n1   直线两个端点的坐标
#   num     采样点个数，如果是None的话，采样点个数和直线长度一样
# 输出
#   list，每个元素是直线上的采样点坐标
def gen_line_sample(n0,n1,num=None):
    if num is None: num=int(round(calc_point_dist(n0,n1)))
    r=np.linspace(0.0,1.0,num)
    return np.array([r*n0[0]+(1.0-r)*n1[0],r*n0[1]+(1.0-r)*n1[1]]).T.tolist(),num


## 功能描述：
#   从hough变换得到参数转成直线两点
# 输入参数：
#   a   直线垂直方向的角度
#   d   原点到直线距离
# 输出：
#   直线的两个点，其中一个是原点到直线的垂直投影,另一点和他距离是1
def hough_param_to_line(a,d):
    x0,y0=d*math.cos(a),d*math.sin(a)
    b=a+math.pi/2.0
    x1,y1=x0+math.cos(b),y0+math.sin(b)
    return (x0,y0),(x1,y1)


####################
## 单元测试
####################
def test_3d_line_fit():
    # 测试3D直线拟合
    x0,y0,z0=4,5,6  # 直线经过的点
    a,b,c=1,2,3     # 直线方向向量
    
    x=np.arange(5)*a+x0
    y=np.arange(5)*b+y0
    z=np.arange(5)*c+z0
    
    if USE_CV_FUNC:
        a_hat,b_hat,c_hat,x0_hat,y0_hat,z0_hat  =line_fit_3d_cv(x,y,z)
    else:
        a_hat,b_hat,c_hat,x0_hat,y0_hat,z0_hat,_=line_fit_3d(x,y,z)
    
    tx=(x-x0_hat)/a_hat
    ty=(y-y0_hat)/b_hat
    tz=(z-z0_hat)/c_hat
    
    print('varify line model: (=0)')
    print(np.linalg.norm(tx-ty))    # 应该接近0
    print(np.linalg.norm(ty-tz))    # 应该接近0
    
    print('varify line model by distance: (=0)')
    print(line_distance_3d(a,b,c,x0,y0,z0,x,y,z))   
    print(line_distance_3d(a_hat,b_hat,c_hat,x0_hat,y0_hat,z0_hat,x,y,z))   


    # 测试3D直线距离计算
    n1=np.array([1,1,-1])   # 注意：n1和n2与[a,b,c]垂直
    n2=np.array([3,0,-1])

    n1=n1/np.linalg.norm(n1)  # 归一化  
    n2=n2/np.linalg.norm(n2)
    
    p1=np.kron(np.arange(5),n1).reshape(5,3)
    p2=np.kron(np.arange(5),n2).reshape(5,3)
    
    print('varify line model by distance:',end=''); print(np.arange(5))
    
    x,y,z=p1[:,0]+x0,p1[:,1]+y0,p1[:,2]+z0
    print(line_distance_3d(a_hat,b_hat,c_hat,x0_hat,y0_hat,z0_hat,x,y,z))   

    x,y,z=p2[:,0]+x0,p2[:,1]+y0,p2[:,2]+z0
    print(line_distance_3d(a_hat,b_hat,c_hat,x0_hat,y0_hat,z0_hat,x,y,z))   


def test_fit_sphere():
    print('---- testing sphere fitting ---')
    
    N=10
    R=np.abs(np.random.rand(1))
    XC,YC,ZC=np.random.rand(1),np.random.rand(1),np.random.rand(1)
    
    pc=make_sphere(R=R,K=N,KS1=1.0,KS2=1.0)
    pc=pc_mov(XC,YC,ZC,pc)   
     
    x0,y0,z0,r=fit_sphere(pc)
    print('center:(%.4f,%.4f,%.4f), r:%.4f'%(x0,y0,z0,r))
    print('Reference')
    print('center:(%.4f,%.4f,%.4f), r:%.4f'%(XC,YC,ZC,R))
    
    
def test_fit_3D_circle():
    print('---- testing 3D circule fitting ---')
    
    # 环参数
    N=10
    R=np.abs(np.random.rand(1))
    XC,YC,ZC=np.random.rand(1),np.random.rand(1),np.random.rand(1)
    
    # 生成环上的点pc
    pha=np.random.rand(N)*np.pi*2
    pcx=np.cos(pha)*R
    pcy=np.sin(pha)*R
    pcz=np.zeros_like(pcx)
    pc=np.concatenate((pcx,pcy,pcz))
    pc.shape=3,len(pcx)
    pc=pc.T
    
    # 随机移动旋转矩阵
    T_rot=np.dot(pc_rotx(np.random.rand(1)*np.pi*2),
                 pc_roty(np.random.rand(1)*np.pi*2))
    T_mov=pc_mov(XC,YC,ZC)
    T=np.dot(T_rot,T_mov)
    
    # 环的空间移动旋转
    pc=pc_trans(T,pc)
     
    xc,yc,zc,r,nx,ny,nz=fit_3D_circule(pc)
    print('center:(%.4f,%.4f,%.4f), r:%.4f, dir:(%.4f,%.4f,%.4f)'%(xc,yc,zc,r,nx,ny,nz))
    print('Reference')
    print('center:(%.4f,%.4f,%.4f), r:%.4f, dir:(%.4f,%.4f,%.4f)'%(XC,YC,ZC,R,T_rot[2,0],T_rot[2,1],T_rot[2,2]))


def test_fit_plane_param():
    print('---- testing plane fitting ---')
    
    # 参数
    KH,KW=3,5
    XC,YC,ZC=np.random.rand(1),np.random.rand(1),np.random.rand(1)
    
    # 生成pc
    pc=make_rectangle(KH=KH,KW=KW)
    
    # 随机移动旋转矩阵
    T_rot=np.dot(pc_rotx(np.random.rand(1)*np.pi*2),
                 pc_roty(np.random.rand(1)*np.pi*2))
    T_mov=pc_mov(XC,YC,ZC)
    T=np.dot(T_rot,T_mov)
    
    # 环的空间移动旋转
    pc=pc_trans(T,pc)
     
    nx,ny,nz,d,px,py,pz=fit_plane_param(pc)
    print('dir:(%.4f,%.4f,%.4f)'%(nx,ny,nz))
    print('Reference')
    print('dir:(%.4f,%.4f,%.4f)'%(T_rot[2,0],T_rot[2,1],T_rot[2,2]))
    print('verify:')
    print(np.max(np.abs(pc[:,0]*nx+pc[:,1]*ny+pc[:,2]*nz+d)))
    print(px*nx+py*ny+pz*nz+d)
    
    
    
if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from depth_cam_tools import *
    
    test_fit_sphere()
    print('')
    
    test_fit_3D_circle()
    print('')
    
    test_fit_plane_param()
