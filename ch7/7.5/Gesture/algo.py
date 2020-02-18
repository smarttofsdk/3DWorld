#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import cv2
import numpy as np
import pylab as plt

sys.path.append('./')
from global_cfg import *

# 向量循环移位
def vector_rotate_right(vec,n):
    sz=len(vec)
    n%=sz
    return vec if n==0 else np.concatenate((vec[sz-n:],vec[:sz-n])) 


def vector_rotate_left(vec,n): return vector_rotate_right(vec,-n)

## 计算轮廓线上的曲率(批量计算)
# 输入：
#   x_contour,y_contour: 轮廓链上的点集（依次排列）
#   L:检测曲率的前向后向轮廓点间距
# 输出：
#   每个轮廓点对应的曲率（夹角cos值）
# 算法描述：概算法计算下图夹角angle
#   x,y
#     (x,y) 
#      /\  angle
#     /  \
#  (xa,ya)\
#          \(xb,yb)
def calc_point_angle(x,y,xa,ya,xb,yb):
    vx1,vy1=x-xa,y-ya
    vx2,vy2=x-xb,y-yb
    tmp=np.sqrt(((vx1**2+vy1**2)*(vx2**2+vy2**2)).astype(np.float))
    tmp[tmp==0]=1.0 # avoid div-0 error
    return (vx1*vx2+vy1*vy2).astype(np.float)/tmp

      
## 检测2D剪影图的轮廓曲率极限点
# 输入：
#   img_bw二值图
# 返回：
#   curvity: 曲率参数，凹陷点为负
#   curvity_type: 弯曲方向检测（True: 凸出点，False：凹陷点）
#   x_contour,y_contour: 轮廓
#   img_contour: 轮廓曲率图
def calc_curvity(img_bw,L=8):
    x_contour,y_contour=calc_contour(img_bw)        # 计算轮廓坐标
    if len(x_contour)==0: # 没有轮廓
        return [],[],[],[],np.zeros((IMG_HGT,IMG_WID))
    
    x_backward =vector_rotate_right(x_contour,L)    # 计算后点
    y_backward =vector_rotate_right(y_contour,L)   
    x_forward  =vector_rotate_left (x_contour,L)    # 计算前点         
    y_forward  =vector_rotate_left (y_contour,L)

    # 计算夹角的余弦
    curvity=calc_point_angle(x_contour ,y_contour ,
                             x_backward,y_backward,
                             x_forward ,y_forward)

    # 区分凹陷点和凸出点
    x_mid=((x_forward+x_backward)/2).astype(int)%IMG_WID
    y_mid=((y_forward+y_backward)/2).astype(int)%IMG_HGT
    curvity_type=img_bw[[y_mid,x_mid]]==1           # 弯曲方向检测结果（True: 凸出点，False：凹陷点）

    # 轮廓曲率强度图（2D)
    img_contour=np.zeros((IMG_HGT,IMG_WID),dtype=np.float)
    img_contour[[y_contour,x_contour]]=curvity 
    
    return curvity,curvity_type,x_contour,y_contour,img_contour


# 计算2D图中连通区域内某给定的点到其他点的最短距离
def shorest_img_path(img,sx,sy):
    img_hgt,img_wid=img.shape
    
    dist  =np.full_like(img,float('inf'),dtype=float)
    prev_x=-np.ones_like(img)
    prev_y=-np.ones_like(img)
    Q=img
    
    dist[sy,sx]=0.0
    xx=np.tile(np.arange(img_wid),img_hgt).reshape(img_hgt,img_wid)
    yy=np.tile(np.arange(img_hgt),img_wid).reshape(img_wid,img_hgt).T
    #xx=np.array(range(img_wid)*img_hgt).reshape(img_hgt,img_wid)
    #yy=np.array(range(img_hgt)*img_wid).reshape(img_wid,img_hgt).T
    
    num=np.sum(Q)
    while num>0:
        idx=np.argmin(dist[Q==1])        
        ux=xx[Q==1].flatten()[idx]
        uy=yy[Q==1].flatten()[idx]
        Q[uy,ux]=0
        num-=1

        for vy,vx in [(uy-1,ux-1),(uy-1,ux),(uy-1,ux+1),(uy,ux-1),(uy,ux),(uy,ux+1),(uy+1,ux-1),(uy+1,ux),(uy+1,ux+1)]:
            #print('uy:%d,ux:%d <--> vy:%d,vx:%d'%(uy,ux,vy,vx))
            if img[vy,vx]==0: continue

            # d=math.sqrt(float((vy-uy)**2+(vx-ux)**2))
            if vy!=uy:
                d=1.0 if vx==ux else 1.4142135623730951
            else:
                d=1.0 if vx!=ux else 0.0
            #print('uy:%d,ux:%d <--> vy:%d,vx:%d  %.3f'%(uy,ux,vy,vx,d))

            alt=dist[uy,ux]+d
            if alt < dist[vy,vx]:               # A shorter path to v has been found
                dist[vy,vx]=alt
                prev_x[vy,vx]=ux
                prev_y[vy,vx]=uy

    return dist, prev_x,prev_y


# 计算3D图中连通区域内某给定的点到其他点的最短距离
# k=p/f, p:相邻像素物理距离,f:焦距
# X=Z*(x-cx)*p/f=Z*(x-cx)*k (cx是图像中心像素位置） 
def shorest_img3d_path(mask,sx,sy,k,th=-1):
    mask_hgt,mask_wid=mask.shape
    cy,cx=float(mask_hgt)*0.5,float(mask_wid)*0.5   # 图像中心像素
    
    # 初始化
    dist  = np.full_like(mask,float('inf'),dtype=float)
    prev_x=-np.ones_like(mask)
    prev_y=-np.ones_like(mask)
    Q=mask
    dist[sy,sx]=0.0
    th2=th*th if th>0 else float('inf')

    xx=np.tile(np.arange(img_wid),img_hgt).reshape(img_hgt,img_wid)
    yy=np.tile(np.arange(img_hgt),img_wid).reshape(img_wid,img_hgt).T
    #xx=np.array(range(mask_wid)*mask_hgt).reshape(mask_hgt,mask_wid)
    #yy=np.array(range(mask_hgt)*mask_wid).reshape(mask_wid,mask_hgt).T
    
    num=np.sum(Q)
    while num>0:
        idx=np.argmin(dist[Q==1])        
        ux=xx[Q==1].flatten()[idx]
        uy=yy[Q==1].flatten()[idx]
        Q[uy,ux]=0
        num-=1
        
        uz=img_dist[uy,ux]
        for vy,vx in [(uy-1,ux-1),(uy-1,ux),(uy-1,ux+1),(uy,ux-1),(uy,ux),(uy,ux+1),(uy+1,ux-1),(uy+1,ux),(uy+1,ux+1)]:
            #print('uy:%d,ux:%d <--> vy:%d,vx:%d'%(uy,ux,vy,vx))
            if mask[vy,vx]==0: continue
            vz=img_dist[vy,vx]
            
            # 计算点u和v之间的3D距离
            d2=(uz-vz)**2
            if d2>th2: continue # 距离差距过大，认为不是连通区域的点云
            if vy!=uy: d2+=((float(vy-cy)*vz-float(uy-cy)*uz)*k)**2
            if vx!=ux: d2+=((float(vx-cx)*vz-float(ux-cx)*uz)*k)**2
            d=math.sqrt(d2)
            
            #print('uy:%d,ux:%d <--> vy:%d,vx:%d  %.3f'%(uy,ux,vy,vx,d))

            alt=dist[uy,ux]+d
            if alt < dist[vy,vx]:               # A shorter path to v has been found
                dist[vy,vx]=alt
                prev_x[vy,vx]=ux
                prev_y[vy,vx]=uy
        
    return dist, prev_x,prev_y


def test_shorest_img_path():
    #              0 1 2 3 4 5 6 7 8 9
    img=np.array([[0,0,0,0,0,0,0,0,0,0],    # 0
                  [0,1,1,1,1,1,1,1,0,0],    # 1
                  [0,0,1,0,0,0,1,1,0,0],    # 2
                  [0,0,0,1,0,0,0,1,0,0],    # 3
                  [0,1,1,1,1,1,1,1,1,0],    # 4
                  [0,0,0,0,0,0,0,0,0,0]])   # 5

    sx,sy=4,4
    dist, prev_x,prev_y=shorest_img_path(img,sx=sx,sy=sy)
    dist1=dist
    dist1[np.isinf(dist1)]=-1
    print('distance to (%d,%d)'%(sx,sy))
    print(dist1)
    
    # trace back
    vx,vy=4,1
    print('trace from (%d,%d) to (%d,%d)'%(vx,vy,sx,sy))
    while vx!=sx or vy!=sy:
        print('(vx,vy):(%d,%d)'%(vx,vy))
        vx1=prev_x[vy,vx]
        vy1=prev_y[vy,vx]
        vx,vy=vx1,vy1


## 色彩变换函数
def bgr_2_ycbcr(img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
def bgr_2_rgb  (img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
def bgr_2_hsv  (img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV   )
def bgr_2_gray (img_bgr): return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY  )


# 肤色检测，Kukharev算法
# 输入OpenCV的BGR图
def skin_det(img_bgr):
    img_ycc = bgr_2_ycbcr(img_bgr)
    (img_Y,img_Cr,img_Cb)= img_ycc[:,:,0],img_ycc[:,:,1],img_ycc[:,:,2]  
    (img_B,img_G ,img_R )= img_bgr[:,:,0],img_bgr[:,:,1],img_bgr[:,:,2]
    
    cond1=np.bitwise_and(img_R>img_G,img_R>img_B)    
    cond2=np.bitwise_or(np.bitwise_and(img_G>img_B,img_R*5-img_G*12+img_B*7 >=0),
                        np.bitwise_and(img_G<img_B,img_R*5+img_G*7 -img_B*12>=0))
    cond3=np.bitwise_and(np.bitwise_and(img_Cr>135,img_Cr<180),
                         np.bitwise_and(img_Cb>85 ,img_Cb<135),
                         img_Y>80)
    skin_det=np.bitwise_and(cond1, cond2,cond3)
    
    img_skin = img_bgr      
    img_skin[~skin_det,:]=0

    return img_skin


# add mark to image
# note that img and mask will be changed by this function call
def img_add_mark(img,mask,xlist,ylist,mark_sz=3,mark_val=1):
    for x,y in zip(xlist,ylist):
        if y>=mark_sz and y+mark_sz<IMG_HGT and x>=mark_sz and x+mark_sz<IMG_WID:
            img[y-mark_sz:y+mark_sz,x-mark_sz:x+mark_sz]=mark_val
            mask[y-mark_sz:y+mark_sz,x-mark_sz:x+mark_sz]=True


# 增加圆圈标记
def img_add_mark_cir(img,mask,xcent,ycent,r=3,val=1):
    # 确定绘制圆周角度步长
    dp=1.0/float(r)
    
    # 计算角度序列
    N=int(np.pi*2/dp)
    a=np.linspace(0,np.pi*2,N)
    
    # 计算圆周坐标
    x=(np.cos(a)*r+xcent).astype(np.int)
    y=(np.sin(a)*r+ycent).astype(np.int)
    
    # 边界截断
    x=np.clip(x,0,IMG_WID-1)
    y=np.clip(y,0,IMG_HGT-1)
    
    # 绘制到img
    img[y,x]=val
    
    # 填充mask
    mask.shape=IMG_HGT,IMG_WID
    mask[y,x]=True
    
    return img,mask
    
    
## 根据距离门限和亮度门限切割手
def hand_cut(img_dep=[],img_amp=[],dmin=-1,dmax=-1,amp_th=-1,mask=None):
    mask_out=np.full((IMG_HGT,IMG_WID),True,dtype=bool) if mask is None else mask.copy()
    
    # 切除过暗的像素
    # if amp_th>=0: mask_out[img_amp<amp_th]=False

    # 切除距离区间外的像素
    if dmin>=0: mask_out[img_dep<dmin]=False
    if dmax>=0: mask_out[img_dep>dmax]=False

    return mask_out


## 计算掌心坐标(x,y)
def get_palm_center(img_bw,return_pix_dist=False):
    # 计算距离图（像素到边界的最近距离）
    pix_dist=cv2.distanceTransform(img_bw,cv2.DIST_L2, maskSize=3).flatten()
    pos=np.argmax(pix_dist)
    x=int(int(pos)%int(IMG_WID))
    y=int(int(pos)/int(IMG_WID))
    if return_pix_dist:
        return (x,y,pix_dist.reshape(IMG_HGT,IMG_WID))
    else:
        return (x,y)


## 计算轮廓图
# 输入是uint8类型的img_bw黑白图，之中只有一个连通区域
def calc_contour(img_bw, gen_mask=False):
    _, contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        if gen_mask: 
            return [],[],[]
        else:
            return [],[]
    
    contours=np.array(contours[0]).flatten()    # 轮廓坐标序列（x/y间隔存放）
    x_contour=contours[0::2]
    y_contour=contours[1::2]
    
    x_contour=x_contour[::FTIP_TRK_CT_DOWN_SAMPLE]
    y_contour=y_contour[::FTIP_TRK_CT_DOWN_SAMPLE]
    
    # 轮廓曲率强度图（2D)
    if gen_mask:
        mask=np.full((IMG_HGT,IMG_WID),False,dtype=bool)
        mask[[y_contour,x_contour]]=True
        return x_contour,y_contour, mask
    else:
        return x_contour,y_contour


# 简易指尖部跟踪器
class ftip_trk_c:
    def __init__(self):
        # 存放过去FTIP_TRK_LIST_SZ帧找到的指尖位置
        self.xlist=np.zeros(FTIP_TRK_LIST_SZ)
        self.ylist=np.zeros(FTIP_TRK_LIST_SZ)
        self.idx=0          # 访问xlist/ylist的索引
        
        # 存放过去FTIP_TRK_LIST_SZ帧找到的指尖位置有效性
        self.valid=np.zeros(FTIP_TRK_LIST_SZ)  
        
        # 常量数组（为提高运算速度）
        self.xidx=np.arange(IMG_WID)    # 0,1,2,...,319
        self.yidx=np.arange(IMG_HGT)    # 0,1,2,...,239
        
        # 当前指尖位置，质量因子和状态字符串
        self.xtip,self.ytip=-1,-1
        self.q0,self.q1=0,0

        # 轨迹跟踪结果
        self.angle=0
        self.decision=''

        # 图示结果
        self.img_view=np.zeros(IMG_SZ)
        self.maks_view=np.zeros(IMG_SZ,dtype=bool)
        
        self.trace_pattern={'type':'none','param':0}

        # 调试图像数据
        self.img_debug=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        
        return

    # 计算指尖位置，输入深度图和强度图
    # 返回指尖位置估计值(xtip=xmean,ytip=ymean)和质量因子(q0,q1)
    # 另外更新手部切割结果（mask_view)
    def calc_tip(self,img_dist,img_amp,mask=None):

        # 初选mask（手部切割）
        mask=hand_cut(img_dep=img_dist,img_amp=img_amp,
                      dmin=FTIP_TRK_DMIN,
                      dmax=FTIP_TRK_DMAX,
                      amp_th=FTIP_TRK_AMP_TH,mask=mask).flatten()
        self.mask_view=mask.copy()

        # 2D化，用于指尖检测
        mask.shape=IMG_HGT,IMG_WID
        img2d=img_dist.copy().reshape(IMG_HGT,IMG_WID)

        # 选出最靠近镜头的FTIP_TRK_DIST_SEL个像素
        pix_sel=img2d[mask]
        if np.size(pix_sel)>FTIP_TRK_DIST_SEL:
            dist_th=np.sort(pix_sel.flatten())[FTIP_TRK_DIST_SEL]
            mask[img2d>dist_th]=False
            
            self.img_debug=mask.copy().astype(np.uint8)*255
        else:
            self.img_debug=np.zeros_like(mask).astype(np.uint8)
            

        # 将选中的最靠近镜头的像素根据距离倒数加权，投影到x和y轴
        xproj=np.sum(mask/(img2d+FTIP_TRK_EPS),axis=0).astype(np.float)
        yproj=np.sum(mask/(img2d+FTIP_TRK_EPS),axis=1).astype(np.float)

        # 归一化（为了方便计算投影中心），归一化的xproj/yproj类似概率密度函数
        xproj/=(np.sum(xproj)+FTIP_TRK_EPS).astype(np.float)
        yproj/=(np.sum(yproj)+FTIP_TRK_EPS).astype(np.float)

        # 计算投影中心（即：指尖位置估计值）
        xmean=np.sum(xproj*self.xidx)
        ymean=np.sum(yproj*self.yidx)

        # 计算质量因子
        q0=np.sum(mask)     # 基于像素数量的质量因子, 越大，质量越好
        q1=math.sqrt(np.sum(xproj*(self.xidx-xmean)**2)+
                     np.sum(yproj*(self.yidx-ymean)**2))    # 基于像素分布方差的质量因子，越大，质量越差

        self.xtip,self.ytip=xmean,ymean
        self.q0,self.q1=q0,q1

        # 返回指尖位置估计值(xmean,ymean)和质量因子
        return xmean,ymean,q0,q1

    # 跟踪指尖，识别轨迹类型以及指尖相对轨迹中心的角度
    # 返回：标记了指尖的图像和对应的手部切割屏蔽
    def calc(self,img_dist,img_amp,mask=None):
        # 存放指尖位置数据的指针+1
        self.idx+=1
        self.idx%=len(self.xlist)
        if self.idx==0: self.fill=True

        # 计算指尖位置，记录得到的坐标
        xtip,ytip,q0,q1=self.calc_tip(img_dist,img_amp,mask)
        self.xlist[self.idx]=xtip
        self.ylist[self.idx]=ytip
        
        # 计算结果有效性判断
        if DEBUG_PRINT: print('q0:%d,q1:%d'%(q0,q1))
        self.valid[self.idx]=0 if q0<FTIP_TRK_Q0_TH or q1>FTIP_TRK_Q1_TH else 1
        self.valid[self.idx]=0 if xtip is math.isnan or ytip is math.isnan else 1
        #self.valid[self.idx]=1
            
        # 有效数据太少，不作进一步分析
        if np.sum(self.valid)<FTIP_TRK_VALID_TH:
            self.img_view=img_dist 
            return (img_dist,self.mask_view)
        
        # 提取所有有效轨迹坐标
        xlist_sel=self.xlist[self.valid==1]
        ylist_sel=self.ylist[self.valid==1]

        # 计算指尖轨迹中心
        xcent=np.mean(xlist_sel)
        ycent=np.mean(ylist_sel)
        
        # 计算x/y坐标轴的指尖轨迹范围
        xr=float(np.max(xlist_sel)-np.min(xlist_sel))/2.0
        yr=float(np.max(ylist_sel)-np.min(ylist_sel))/2.0
        
        # 计算45度倾斜坐标轴的指尖轨迹范围
        xrot=np.float32((xlist_sel-xcent)-(ylist_sel-ycent))*0.707106781
        yrot=np.float32((xlist_sel-xcent)+(ylist_sel-ycent))*0.707106781
        
        xr_rot=float(np.max(xrot)-np.min(xrot))/2.0
        yr_rot=float(np.max(yrot)-np.min(yrot))/2.0
        
        # 计算当前指尖相对于中心的旋转角度
        angle=-math.atan2(ytip-ycent,xtip-xcent)*180.0/np.pi

        # 计算轨迹偏离圆周的相对量
        cir_err=1.0-np.min(((xlist_sel-xcent)**2+(ylist_sel-ycent)**2)/((xr+yr)**2/4))
        
        # decision
        decision='none'
        if xr<FTIP_TRK_POINT_TH and yr<FTIP_TRK_POINT_TH                : decision='point'
        elif xr>yr*FTIP_TRK_HLINE_TH                                    : decision='horizontal'
        elif yr>xr*FTIP_TRK_VLINE_TH                                    : decision='vertical'
        # elif xr_rot>yr_rot*FTIP_TRK_LINE45_TH                           : decision='line 45'
        # elif yr_rot>xr_rot*FTIP_TRK_LINE135_TH                          : decision='line 135'
        elif yr<xr*FTIP_TRK_CIRCLE_TH:
            if cir_err<FTIP_TRK_CIRCLE_ERR: 
                decision='circle'
            else:
                print('cir_err:%.4f'%cir_err)

        if DEBUG_PRINT:
            print('%s,angle:%.1f,xtip:%.1f,ytip:%.1f,xr:%.1f,yr:%.1f,xr/yr:%.1f,yr/xr:%.1f,sum(valid):%d'%
                   (decision,angle,xtip,ytip,xr,yr,xr/yr,yr/xr,np.sum(self.valid)))

        self.decision=decision
        self.angle=angle
        
        self.img_view=img_dist.copy().reshape(IMG_HGT,IMG_WID)
        
        # img_add_mark(self.img_view,self.mask_view.reshape(IMG_HGT,IMG_WID),[int(xtip)],[int(ytip)],FTIP_TRK_MARK_SZ,0.0)

        self.img_view.shape=IMG_SZ
        self.mask_view.shape=IMG_SZ
        
        self.trace_pattern={'type':'none','param':0}
        
        if   self.valid[self.idx]==0    : self.trace_pattern={'type':'none'    ,'param':0}
        elif self.decision=='circle'    : self.trace_pattern={'type':'circle'  ,'param':angle}
        elif self.decision=='horizontal': self.trace_pattern={'type':'h-line'  ,'param':(xtip-xcent)/xr}
        elif self.decision=='vertical'  : self.trace_pattern={'type':'v-line'  ,'param':(ytip-ycent)/yr}
        elif self.decision=='line 45'   : self.trace_pattern={'type':'line-45' ,'param':(xtip-xcent)/xr}
        elif self.decision=='line 135'  : self.trace_pattern={'type':'line-135','param':(xtip-xcent)/xr}
        elif self.decision=='point'     : self.trace_pattern={'type':'point'   ,'param':0}
        
        return (self.img_view,self.mask_view)


# 多指尖检测器
class multi_ftip_det_c:
    def __init__(self):    
        # 存放显示图像   
        self.img_view =np.zeros(IMG_SZ)
        self.mask_view=np.zeros(IMG_SZ,dtype=bool)
        
        # 去噪核(会被多次使用）
        self.denoise_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(FTIP_TRK_DENOISE_TH,FTIP_TRK_DENOISE_TH))  
        
        # 存放多指尖跟踪结果
        self.xtip_list=[]       # 指尖坐标
        self.ytip_list=[]
        self.xfinger_line=[]    # 手指中心线
        self.yfinger_line=[]

        # 存放手掌参数
        self.x_palm,self.y_palm,self.palm_area,self.palm_r=0,0,0,0
        
        # 存放计算得到的中间结果
        self.img_hand=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        self.img_palm=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        
        # 调试图像数据
        self.img_debug=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        
        return

    # 计算轮廓弯曲，输入深度图和强度图
    # 检出待选指尖坐标
    def calc(self,img_dist,img_amp,mask=[]):
        # 手部切割,得到手的2D剪影（二值图）
        img_hand=hand_cut(img_dep=img_dist,         # 深度图
                          img_amp=img_amp,          # 亮度图
                          dmin=FTIP_TRK_DMIN,       # 距离切割门限
                          dmax=FTIP_TRK_DMAX,   
                          amp_th=FTIP_TRK_AMP_TH,   # 亮度切割门限
                          mask=mask).astype(np.uint8)
        img_hand.shape=IMG_HGT,IMG_WID
        # cv2.imshow("hand",img_hand*255)
        # cv2.waitKey(1)
        # 去除img_hand中噪点
        #img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_OPEN ,self.denoise_kernel)  
        img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_CLOSE,self.denoise_kernel)
        
        self.img_debug=img_hand.copy()

        # 计算掌心位置，同时输出像素到边界的距离
        x_palm,y_palm,pix_dist=get_palm_center(img_hand,True)
        
        # 手掌面切割，计算手掌参数
        # 可以简化，只用palm_r=pix_dist[y_palm,x_palm]
        img_palm=cv2.morphologyEx(img_hand,
                                  cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)))
        palm_area=np.sum(img_palm)  # 计算手掌面积
        palm_r1=np.sqrt(float(palm_area)/np.pi/2.0)*1.2  # 计算手掌半径
        palm_r2=pix_dist[y_palm,x_palm]
        palm_r=max(palm_r1,palm_r2)
        
        ## 计算轮廓和曲率
        curvity,curvity_type,x_contour,y_contour,img_contour=calc_curvity(img_hand)
        if len(curvity)==0: # 没有轮廓
            self.img_view=img_hand.flatten()
            self.img_mask=img_hand.flatten()
            return (self.img_view,self.mask_view)
        num_contour=len(x_contour)  # 轮廓点数量
         
        # 找出可能的指尖区域
        w_th  =np.bitwise_and(curvity>FTIP_TRK_BEND_TH,curvity_type)    # 通过门限截出可能的指尖区并只保留凸出区域
        w_th = w_th.astype(np.uint8)
        
        # 对w_th循环滤波，结果存放于w_det (去除孤立点和填充孤立洞)
        w_det=(w_th+vector_rotate_left (w_th,1)+vector_rotate_right(w_th,1))>1   
        w_det=w_det.astype(np.uint8)
        
        # 指尖点定位
        idx_start,idx_end=[],[]
        for n in range(num_contour):    # w_det上的连通域(连1区域)起始和结束位置检测
            if w_det[n]>w_det[n-1]: idx_start.append(n)             # 轮廓上连1区域起始位置
            if w_det[n]>w_det[(n+1)%num_contour]: idx_end.append(n) # 轮廓上连1区域结束位置

        if len(idx_start)>0:
            # 轮廓上连1区域中间位置计算
            if idx_end[0]<idx_start[0]:         # 确保连1区域开始位置小于结束位置
                idx_tip=np.array([int((idx_start[n-1]+idx_end[n])/2)  for n in range(len(idx_start))])
                idx_tip[0]=(idx_tip[0]+num_contour/2)%num_contour
                idx_tip=np.array(idx_tip)       # idx_mid是指尖在轮廓数据序列中的位置
            else:
                idx_tip=((np.array(idx_start)+np.array(idx_end))/2).astype(int) # idx_mid是指尖在轮廓数据序列中的位置
            # 找到并保存指尖位置
            xtip_list,ytip_list=x_contour[idx_tip],y_contour[idx_tip]
        else:   # 没有指尖区间
            xtip_list,ytip_list,idx_tip=[],[],[]
        
        # 进一步筛选指尖（距离指尖更大距离的轮廓上两边的点的夹角检验）
        if len(xtip_list)>0:
            Q=20
            x_backward,y_backward=x_contour[(idx_tip-Q)%num_contour],y_contour[(idx_tip-Q)%num_contour]   # 指尖对应轮廓位置的后向点
            x_forward ,y_forward =x_contour[(idx_tip+Q)%num_contour],y_contour[(idx_tip+Q)%num_contour]   # 指尖对应轮廓位置的前向点
            tip_sel = calc_point_angle(xtip_list ,ytip_list , # 计算夹角余弦是否超过门限FTIP_TRK_BEND_TH2
                                       x_backward,y_backward,
                                       x_forward ,y_forward)>FTIP_TRK_BEND_TH2
            xtip_list=xtip_list[tip_sel]
            ytip_list=ytip_list[tip_sel]
            idx_tip  =idx_tip  [tip_sel]
        
        # 计算手指中心线(会进一步过滤手指）
        xtip_list_new=[]
        ytip_list_new=[]
        idx_tip_new  =[]
        
        xfinger_line=[]
        yfinger_line=[]
        alpha=np.linspace(0,1.0,40) # 用于计算指尖中心线，用于计算两点连线，插值40个点
        for xtip,ytip,idx in zip(xtip_list,ytip_list,idx_tip):  # 遍历候选指尖点
            # 轮廓上，指尖位置前后向点集序号
            idx_backward=(idx+np.arange(-1,-40,-1))%num_contour 
            idx_forward =(idx+np.arange( 1, 40, 1))%num_contour
            
            # 点集需要在遇到轮廓凹陷处截断
            tmp1 = np.bitwise_and(~curvity_type[idx_backward], curvity[idx_backward]>FTIP_TRK_BEND_TH)  # 后向大曲率凹陷点
            tmp2 = np.bitwise_and(~curvity_type[idx_forward ], curvity[idx_forward ]>FTIP_TRK_BEND_TH)  # 前向大曲率凹陷点
            cut_pos = np.where(np.bitwise_or(tmp1,tmp2))[0]
            if len(cut_pos): 
                idx_cut=cut_pos[0]  # 截断点集序号
                idx_backward=idx_backward[:idx_cut]
                idx_forward =idx_forward [:idx_cut]
            
            # 该待处理指尖轮廓邻近区域无效，忽略之（无效指尖）
            if len(idx_backward)==0 or len(idx_forward)==0:
                continue;

            # 指尖在轮廓上的后向点集
            xtip_backward=x_contour[idx_backward]
            ytip_backward=y_contour[idx_backward] 
            
            # 指尖在轮廓上的前向点集
            xtip_forward =x_contour[idx_forward] 
            ytip_forward =y_contour[idx_forward]

            # 计算中心线（简单方式）
            #xline=((xtip_backward+xtip_forward)/2).astype(int)      
            #yline=((ytip_backward+ytip_forward)/2).astype(int)
            
            # 基于距离图计算中心线位置
            xline,yline=[],[]
            k=0
            for xb,yb,xf,yf in zip(xtip_backward,ytip_backward,xtip_forward,ytip_forward):  
                x=(xb*alpha+(1.0-alpha)*xf).astype(np.int)  # 构成直线段(xb,yb),(xf,yf)上的离散采样点
                y=(yb*alpha+(1.0-alpha)*yf).astype(np.int)
                idx=np.argmax(pix_dist[y,x])# 找到直线段上最大边距像素序号
                yc,xc=y[idx],x[idx]         # 直线段上最大边距像素坐标
                d=pix_dist[yc,xc]           # 直线段上最大边距像素的边距值
                k+=1
                
                if (k>20) and ((d>16) or (d<8)): 
                    continue    # 除指尖外的手指中线的像素边距超出手指宽度范围(半径16～8）
                if ((xc-x_palm)**2+(yc-y_palm)**2)<palm_r**2: 
                    continue    # 中心线进入掌心区域
                
                xline.append(xc)
                yline.append(yc)
            xline=np.array(xline)
            yline=np.array(yline)
            
            # 根据长度（中心线两个端点距离）确定有效手指线
            if len(xline)>2:
                if (xline[0]-xline[-1])**2+(yline[0]-yline[-1])**2>20**2:
                    xfinger_line.append(xline)  # 保存有效中心线
                    yfinger_line.append(yline)
                    
                    xtip_list_new.append(xtip)  # 保存有效指尖坐标
                    ytip_list_new.append(ytip)
                    idx_tip_new.append(idx)
        
        # 更新指尖列表
        xtip_list=xtip_list_new
        ytip_list=ytip_list_new
        idx_tip  =idx_tip_new   # 指尖在轮廓数据序列中的位置

        # 保存最终结果
        self.xtip_list=xtip_list        # 指尖坐标
        self.ytip_list=ytip_list
        self.xfinger_line=xfinger_line  # 手指中心线
        self.yfinger_line=yfinger_line
        
        self.x_palm=x_palm
        self.y_palm=y_palm
        self.palm_area=palm_area
        self.palm_r=palm_r
        
        self.img_hand=img_hand
        self.img_palm=img_palm
        
        # 显示内容处理
        mask_view=~(img_contour==0)  # 只显示轮廓区域
        img_view = -img_contour.reshape(IMG_HGT,IMG_WID) # 显示轮廓曲率
        
        # 添加手指标记
        if len(xtip_list)>0:
            for xtip,ytip,xline,yline in zip(xtip_list,ytip_list,xfinger_line,yfinger_line):
                img_add_mark_cir(img_view,mask_view,xtip,ytip)
                img_view [yline,xline]=0
                mask_view[yline,xline]=True
        
        # 画上掌心标记和手掌拟合的圆
        img_add_mark(img_view,mask_view,[x_palm],[y_palm])
        if palm_r>0: img_add_mark_cir(img_view,mask_view,x_palm,y_palm,palm_r)
        
        # 保存需要显示的内容到类变量
        self.img_view =img_view.flatten()
        self.mask_view=mask_view.flatten()
        
        return (self.img_view,self.mask_view)

## 红外图像处理（轮廓匹配）
class IR_algo_c:
    def __init__(self):    
        # 存放显示图像   
        self.img_view =np.zeros(IMG_SZ)
        self.mask_view=np.zeros(IMG_SZ,dtype=bool)
        
        # 去噪核(会被多次使用）
        self.denoise_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(FTIP_TRK_DENOISE_TH,FTIP_TRK_DENOISE_TH))  
        
        # 存放多指尖跟踪结果
        self.xtip_list=[]       # 指尖坐标
        self.ytip_list=[]
        self.xfinger_line=[]    # 手指中心线
        self.yfinger_line=[]

        # 存放手掌参数
        self.x_palm,self.y_palm,self.palm_area,self.palm_r=0,0,0,0
        
        # 存放计算得到的中间结果
        self.img_hand=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        self.img_palm=np.zeros((IMG_HGT,IMG_WID),dtype=np.uint8)
        
        # 轮廓模板
        finger_two={'x':np.array([178, 180, 182, 184, 186, 188, 190, 191, 193, 195, 197, 199, 201, 202, 204, 205, 207, 208, 209, 211, 213, 215, 217, 219, 221, 222, 223, 223, 222, 221, 219, 217, 215, 213, 211, 210, 208, 206, 205, 203, 201, 200, 198, 196, 194, 192, 191, 189, 187, 185, 183, 181, 179, 177, 176, 174, 172, 170, 168, 166, 164, 162, 162, 161, 161, 161, 161, 161, 162, 162, 162, 162, 163, 163, 164, 164, 165, 165, 165, 166, 167, 167, 168, 168, 169, 169, 170, 170, 171, 170, 169, 168, 166, 164, 162, 160, 158, 157, 156, 155, 154, 154, 153, 153, 152, 151, 151, 150, 150, 149, 148, 147, 147, 146, 146, 145, 144, 144, 143, 142, 142, 141]),
                    'y':np.array([159, 157, 156, 155, 154, 152, 150, 148, 147, 145, 143, 142, 140, 138, 137, 135, 133, 131, 130, 129, 127, 125, 124, 122, 120, 118, 116, 114, 112, 110, 109, 108, 108, 109, 110, 111, 112, 114, 115, 117, 118, 120, 121, 122, 124, 125, 127, 128, 130, 132, 133, 135, 137, 138, 140, 141, 142, 144, 145, 146, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 85, 84, 84, 84, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137])}
    
        # 计算轮廓匹配误差投影矩阵Pe
        #   G=[c 1]             c是模板轮廓上的点（复数表示）对应列向量，1是指全1构成的列向量
        #   Pc=G*inv(G'*G)*G'   轮廓投影矩阵，将模板投影到被匹配轮廓（c2=Pc*c1是模板轮廓c最接近c1的投影）
        #   Pe=Pc-I             误差投影矩阵
        #   w_opt=Q*c1          w[0]:缩放旋转因子，w[1]:平移因子
        c=finger_two['x']+finger_two['y']*1j
        N=len(c)
        c.shape=N,1
        G=np.hstack((c,np.ones((N,1))))
        G_h=G.conj().T
        Pc=np.dot(G,np.dot(np.linalg.pinv(np.dot(G_h,G)),G_h))
        Pe=Pc-np.eye(N)
        
        self.Pc=Pc
        self.Pe=Pe
        self.N =N
        return

    # 根据轮廓弯曲找到手指，并计算手指中心线
    # 输入深度图和强度图
    # 检出待选指尖坐标
    def calc_finger(self,img_amp,mask=[]):
        # 手部切割,得到手的2D剪影（二值图）
        img_hand=hand_cut(img_dep=[],img_amp=img_amp.flatten(),
                          amp_th=ALGO_CONTOUR_MATCH_IR_AMP_TH,
                          mask=mask)
        img_hand=img_hand.reshape(IMG_HGT,IMG_WID).astype(np.uint8)
        
        # 去除img_hand中噪点
        img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_OPEN ,self.denoise_kernel)  
        img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_CLOSE,self.denoise_kernel)
        
        # 计算掌心位置，同时输出像素到边界的距离
        x_palm,y_palm,pix_dist=get_palm_center(img_hand,True)
        
        # 手掌面切割，计算手掌参数
        # 可以简化，只用palm_r=pix_dist[y_palm,x_palm]
        img_palm=cv2.morphologyEx(img_hand,
                                  cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)))
        palm_area=np.sum(img_palm)  # 计算手掌面积
        palm_r1=np.sqrt(float(palm_area)/np.pi/2.0)*1.2  # 计算手掌半径
        palm_r2=pix_dist[y_palm,x_palm]
        palm_r=max(palm_r1,palm_r2)
        
        ## 计算轮廓和曲率
        curvity,curvity_type,x_contour,y_contour,img_contour=calc_curvity(img_hand)
        if len(curvity)==0: # 没有轮廓
            self.img_view=img_hand.flatten()
            self.img_mask=img_hand.flatten()
            return (self.img_view,self.mask_view)
        num_contour=len(x_contour)  # 轮廓点数量
         
        # 找出可能的指尖区域
        w_th  =np.bitwise_and(curvity>FTIP_TRK_BEND_TH,curvity_type)    # 通过门限截出可能的指尖区并只保留凸出区域
        w_th = w_th.astype(np.uint8)
        
        # 对w_th循环滤波，结果存放于w_det (去除孤立点和填充孤立洞)
        w_det=(w_th+vector_rotate_left (w_th,1)+vector_rotate_right(w_th,1))>1   
        w_det=w_det.astype(np.uint8)
        
        # 指尖点定位
        idx_start,idx_end=[],[]
        for n in range(num_contour):    # w_det上的连通域(连1区域)起始和结束位置检测
            if w_det[n]>w_det[n-1]: idx_start.append(n)             # 轮廓上连1区域起始位置
            if w_det[n]>w_det[(n+1)%num_contour]: idx_end.append(n) # 轮廓上连1区域结束位置

        if len(idx_start)>0:
            # 轮廓上连1区域中间位置计算
            if idx_end[0]<idx_start[0]:         # 确保连1区域开始位置小于结束位置
                idx_tip=np.array([int((idx_start[n-1]+idx_end[n])/2)  for n in range(len(idx_start))])
                idx_tip[0]=(idx_tip[0]+num_contour/2)%num_contour
                idx_tip=np.array(idx_tip)       # idx_mid是指尖在轮廓数据序列中的位置
            else:
                idx_tip=((np.array(idx_start)+np.array(idx_end))/2).astype(int) # idx_mid是指尖在轮廓数据序列中的位置
            # 找到并保存指尖位置
            xtip_list,ytip_list=x_contour[idx_tip],y_contour[idx_tip]
        else:   # 没有指尖区间
            xtip_list,ytip_list,idx_tip=[],[],[]
        
        # 删除屏幕边沿的指尖（手臂图像截断造成的“假尖角”）
        if len(xtip_list)>0:
            tip_sel=np.bitwise_and(np.bitwise_and(xtip_list<IMG_WID*0.9,xtip_list>IMG_WID*0.1),
                                   np.bitwise_and(ytip_list<IMG_HGT*0.9,ytip_list>IMG_HGT*0.1))
            xtip_list=xtip_list[tip_sel]
            ytip_list=ytip_list[tip_sel]
            idx_tip  =idx_tip  [tip_sel]
        
        
        # 进一步筛选指尖（距离指尖更大距离的轮廓上两边的点的夹角检验）
        if len(xtip_list)>0:
            Q=20
            x_backward,y_backward=x_contour[(idx_tip-Q)%num_contour],y_contour[(idx_tip-Q)%num_contour]   # 指尖对应轮廓位置的后向点
            x_forward ,y_forward =x_contour[(idx_tip+Q)%num_contour],y_contour[(idx_tip+Q)%num_contour]   # 指尖对应轮廓位置的前向点
            tip_sel = calc_point_angle(xtip_list ,ytip_list , # 计算夹角余弦是否超过门限FTIP_TRK_BEND_TH2
                                       x_backward,y_backward,
                                       x_forward ,y_forward)>FTIP_TRK_BEND_TH2
            xtip_list=xtip_list[tip_sel]
            ytip_list=ytip_list[tip_sel]
            idx_tip  =idx_tip  [tip_sel]
        
        # 计算手指中心线(会进一步过滤手指）
        xtip_list_new=[]
        ytip_list_new=[]
        idx_tip_new  =[]
        
        xfinger_line=[]
        yfinger_line=[]
        alpha=np.linspace(0,1.0,40) # 用于计算指尖中心线，用于计算两点连线，插值40个点
        for xtip,ytip,idx in zip(xtip_list,ytip_list,idx_tip):  # 遍历候选指尖点
            # 轮廓上，指尖位置前后向点集序号
            idx_backward=(idx+np.arange(-1,-40,-1))%num_contour 
            idx_forward =(idx+np.arange( 1, 40, 1))%num_contour
            
            # 点集需要在遇到轮廓凹陷处截断
            tmp1 = np.bitwise_and(~curvity_type[idx_backward], curvity[idx_backward]>FTIP_TRK_BEND_TH)  # 后向大曲率凹陷点
            tmp2 = np.bitwise_and(~curvity_type[idx_forward ], curvity[idx_forward ]>FTIP_TRK_BEND_TH)  # 前向大曲率凹陷点
            cut_pos = np.where(np.bitwise_or(tmp1,tmp2))[0]
            if len(cut_pos): 
                idx_cut=cut_pos[0]  # 截断点集序号
                idx_backward=idx_backward[:idx_cut]
                idx_forward =idx_forward [:idx_cut]
            
            # 该待处理指尖轮廓邻近区域无效，忽略之（无效指尖）
            if len(idx_backward)==0 or len(idx_forward)==0:
                continue;

            # 指尖在轮廓上的后向点集
            xtip_backward=x_contour[idx_backward]
            ytip_backward=y_contour[idx_backward] 
            
            # 指尖在轮廓上的前向点集
            xtip_forward =x_contour[idx_forward] 
            ytip_forward =y_contour[idx_forward]

            # 计算中心线（简单方式）
            #xline=((xtip_backward+xtip_forward)/2).astype(int)      
            #yline=((ytip_backward+ytip_forward)/2).astype(int)
            
            # 基于距离图计算中心线位置
            xline,yline=[],[]
            k=0
            for xb,yb,xf,yf in zip(xtip_backward,ytip_backward,xtip_forward,ytip_forward):  
                x=(xb*alpha+(1.0-alpha)*xf).astype(np.int)  # 构成直线段(xb,yb),(xf,yf)上的离散采样点
                y=(yb*alpha+(1.0-alpha)*yf).astype(np.int)
                idx=np.argmax(pix_dist[y,x])# 找到直线段上最大边距像素序号
                yc,xc=y[idx],x[idx]         # 直线段上最大边距像素坐标
                d=pix_dist[yc,xc]           # 直线段上最大边距像素的边距值
                k+=1
                
                if (k>20) and ((d>16) or (d<8)): 
                    continue    # 除指尖外的手指中线的像素边距超出手指宽度范围(半径16～8）
                if ((xc-x_palm)**2+(yc-y_palm)**2)<palm_r**2: 
                    continue    # 中心线进入掌心区域
                
                xline.append(xc)
                yline.append(yc)
            xline=np.array(xline)
            yline=np.array(yline)
            
            # 根据长度（中心线两个端点距离）确定有效手指线
            if len(xline)>2:
                if (xline[0]-xline[-1])**2+(yline[0]-yline[-1])**2>20**2:
                    xfinger_line.append(xline)  # 保存有效中心线
                    yfinger_line.append(yline)
                    
                    xtip_list_new.append(xtip)  # 保存有效指尖坐标
                    ytip_list_new.append(ytip)
                    idx_tip_new.append(idx)
        
        # 更新指尖列表
        xtip_list=xtip_list_new
        ytip_list=ytip_list_new
        idx_tip  =idx_tip_new   # 指尖在轮廓数据序列中的位置

        # 保存最终结果
        self.xtip_list=xtip_list        # 指尖坐标
        self.ytip_list=ytip_list
        self.xfinger_line=xfinger_line  # 手指中心线
        self.yfinger_line=yfinger_line
        
        self.x_palm=x_palm
        self.y_palm=y_palm
        self.palm_area=palm_area
        self.palm_r=palm_r
        
        self.img_hand=img_hand
        self.img_palm=img_palm
        
        # 显示内容处理
        mask_view=~(img_contour==0)  # 只显示轮廓区域
        img_view = -img_contour.reshape(IMG_HGT,IMG_WID) # 显示轮廓曲率
        
        # 添加手指标记
        if len(xtip_list)>0:
            for xtip,ytip,xline,yline in zip(xtip_list,ytip_list,xfinger_line,yfinger_line):
                img_add_mark_cir(img_view,mask_view,xtip,ytip)
                img_view [yline,xline]=0
                mask_view[yline,xline]=True
        
        # 画上掌心标记和手掌拟合的圆
        img_add_mark(img_view,mask_view,[x_palm],[y_palm])
        if palm_r>0: img_add_mark_cir(img_view,mask_view,x_palm,y_palm,palm_r)
        
        # 保存需要显示的内容到类变量
        self.img_view =img_view.flatten()
        self.mask_view=mask_view.flatten()
        
        return (self.img_view,self.mask_view)

    # 计算“二指竖起”的轮廓匹配，输入红外强度图
    # 输入img_amp是0～255的浮点数（320x240）
    def calc_match_contour(self,img_amp,mask=[]):
        # 截取手
        img_hand=hand_cut(img_dep=[],img_amp=img_amp.flatten(),
                          amp_th=ALGO_CONTOUR_MATCH_IR_AMP_TH,
                          mask=mask)
        img_hand=img_hand.reshape(IMG_HGT,IMG_WID).astype(np.uint8)
        
        # 去除img_hand中噪点
        img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_OPEN ,self.denoise_kernel)  
        img_hand=cv2.morphologyEx(img_hand,cv2.MORPH_CLOSE,self.denoise_kernel)
        
        # 计算轮廓
        x_contour,y_contour,img_contour=calc_contour(img_hand,gen_mask=True)
        M=len(x_contour)            # 轮廓尺寸
        if M==0: return [],[]       # 没有轮廓
        
        N=self.N                    # 模板尺寸
        if M<N: return [],[]        # 轮廓比模板小 
        
        # 计算投影矩阵和轮廓的卷积（numpy会根据效率自动调用FFT）
        c1=(x_contour+y_contour*1j).flatten()       # 轮廓采样点复数化
        c1_ext=np.flipud(np.hstack((c1,c1[:N-1])))  # 循环扩展（为了循环卷积）
        e_all=np.zeros((N,M))*1j                    # 每一列存放一个轮廓滑动窗口的匹配误差向量
        for n in range(N):  # OPTME! 预先计算Pe的FFT，这样能减少一次FFT
            e_all[n,:]=np.convolve(self.Pe[n,:],c1_ext,mode='valid')
        e_vec=np.mean(np.abs(e_all)**2,axis=0)      # 每个元素对应一个轮廓滑动窗口的匹配误差数值
        
        # 找到最优匹配区域
        idx=np.argmin(e_vec)
        if ALGO_CONTOUR_MATCH_PRINT_EMIN:
            print('Min. contour match error:%f'%e_vec[idx]) # 打印匹配误差
        
        x_sel=vector_rotate_right(x_contour,idx)[:N]    # 提取被匹配的轮廓区间坐标
        y_sel=vector_rotate_right(y_contour,idx)[:N]
        c_sel=(x_sel+y_sel*1j).flatten()
        
        # 计算模板在匹配上的轮廓区间的投影
        if ALGO_CONTOUR_MATCH_DRAW_PRJ:
            c_prj=np.dot(self.Pc,c_sel)
            x_prj=np.clip(np.round(np.real(c_prj)).astype(int),0,IMG_WID-1)
            y_prj=np.clip(np.round(np.imag(c_prj)).astype(int),0,IMG_HGT-1)
        
        # 匹配内容标记到图imag_match上
        img_match=np.zeros((IMG_HGT,IMG_WID))
        img_match[y_sel,x_sel] =0.5
        if ALGO_CONTOUR_MATCH_DRAW_PRJ:
            img_match[y_prj,x_prj] =0.2

        # 对于匹配误差门限内的轮廓，可以绘制在屏幕上
        img_view=img_contour.astype(np.float)
        if np.sqrt(e_vec[idx])<ALGO_CONTOUR_MATCH_TH:  # FIXME! 匹配误差门限需要自适应归一化
            img_view += img_match.astype(np.float)
        
        # 保存需要显示的内容到类变量
        self.img_view = img_view.flatten()
        self.mask_view= self.img_view!=0
        self.img_hand = img_hand
        
        return (self.img_view,self.mask_view)

####################
# 以下代码用于算法调试
####################
if __name__ == '__main__':
    # 轮廓模板
    finger_two={'x':np.array([178, 180, 182, 184, 186, 188, 190, 191, 193, 195, 197, 199, 201, 202, 204, 205, 207, 208, 209, 211, 213, 215, 217, 219, 221, 222, 223, 223, 222, 221, 219, 217, 215, 213, 211, 210, 208, 206, 205, 203, 201, 200, 198, 196, 194, 192, 191, 189, 187, 185, 183, 181, 179, 177, 176, 174, 172, 170, 168, 166, 164, 162, 162, 161, 161, 161, 161, 161, 162, 162, 162, 162, 163, 163, 164, 164, 165, 165, 165, 166, 167, 167, 168, 168, 169, 169, 170, 170, 171, 170, 169, 168, 166, 164, 162, 160, 158, 157, 156, 155, 154, 154, 153, 153, 152, 151, 151, 150, 150, 149, 148, 147, 147, 146, 146, 145, 144, 144, 143, 142, 142, 141]),
                'y':np.array([159, 157, 156, 155, 154, 152, 150, 148, 147, 145, 143, 142, 140, 138, 137, 135, 133, 131, 130, 129, 127, 125, 124, 122, 120, 118, 116, 114, 112, 110, 109, 108, 108, 109, 110, 111, 112, 114, 115, 117, 118, 120, 121, 122, 124, 125, 127, 128, 130, 132, 133, 135, 137, 138, 140, 141, 142, 144, 145, 146, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 85, 84, 84, 84, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137])}

    # 计算轮廓匹配误差投影矩阵Pe
    c=finger_two['x']+finger_two['y']*1j
    N=len(c)
    c.shape=N,1
    G=np.hstack((c,np.ones((N,1))))
    G_h=G.conj().T
    Pe=np.dot(G_h,G)
    Pe=np.linalg.pinv(Pe)
    Pe=np.dot(Pe,G_h)
    Pe=np.dot(G,Pe)
    Pe=Pe-np.eye(N)
    
    IR_FNAME='opencam_video_f32_1.bin'    
    fid=open(IR_FNAME,'rb')
        
    for k in range(300):
        # 读取图像
        img_f32=np.fromfile(fid,dtype=np.float32,count=76800)
        
        # 截取手
        img_hand=(img_f32>0.2).reshape(IMG_HGT,IMG_WID).astype(np.uint8)
        img_hand=np.fliplr(img_hand).astype(np.uint8)
        
        # 计算轮廓和曲率
        curvity,curvity_type,x_contour,y_contour,img_contour=calc_curvity(img_hand)
        mask_contour=(img_contour!=0)
        
        if len(curvity)>0:
            num_contour=len(x_contour)  # 轮廓点数量
            plt.subplot(3,1,1)
            plt.imshow(mask_contour)
    
            # 计算投影矩阵和轮廓的卷积，M:轮廓尺寸，N：模板尺寸
            M=len(x_contour)    
            if M<N: continue
            
            c1=(x_contour+y_contour*1j).flatten()
            c1_ext=np.flipud(np.hstack((c1,c1[:N-1])))
            
            e_all=np.zeros((N,M))*1j
            for n in range(N):
                rc1=np.convolve(Pe[n,:],c1_ext,mode='valid')
                e_all[n,:]=rc1
            e_vec=np.sum(np.abs(e_all)**2,axis=0)
            
            # 找到最优匹配区域
            idx=np.argmin(e_vec)
            x_rot=vector_rotate_right(x_contour,idx)
            y_rot=vector_rotate_right(y_contour,idx)
            img_match=np.zeros((IMG_HGT,IMG_WID))
            img_match[y_rot[:N],x_rot[:N]]=1
            
            plt.subplot(3,1,2)
            plt.imshow(mask_contour.astype(np.float)*0.5+img_match.astype(np.float)*0.5)
            
            plt.subplot(3,1,3)
            plt.plot(e_vec)
            
            plt.show()
    
    
