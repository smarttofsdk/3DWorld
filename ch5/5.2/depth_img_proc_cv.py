#!/usr/bin/python3
# -*- coding: utf-8 -*-

##############################
#   深度图处理库(CV图像接口）    #
#         V20181030          #
#----------------------------#
# 20180201: 增加飞散点滤波器class
# 20180321: 增加背景累积计算程序
# 20180412：恢复原先的测试代码
# 20180802: 将calc_bg改成calc_bg_from_file
#           增加calc_bg，从cam设备读取图像并累积
# 20180814：calc_bg函数增加了强度图平均
# 20180829: calc_bg加入skip功能
# 20181030: 改为cv_viewer接口
##############################

# 编码规则说明
# 1. function——实现不具有记忆状态和可配置参数的图像处理功能
# 2. class——实现带有配置参数和状态变量的图像处理功能
# 3. class中set_param()函数创建所有可配置参数
#    1) 运行时刻不可修改或者需要调用成员函数修改的可配置参数用大写字母单词标识
#    2) 运行时刻可随时修改的可配置参数用小写字母单词标识
# 4. class对象构建有两种
#   1) 使用缺省参数设置
#       obj=class_of_obj(xxx,xxx)               # 创建对象，使用缺省参数设置，并完成状态变量和构件初始化
#   2) 手动设置参数和初始化状态变量, 代码序列如下：
#       obj=class_of_obj(init=False)            # 创建空对象
#       obj.set_param(xxx)                      # 调用缺省参数设置
#       obj.xxx=xxx                             # 用户修改缺省参数设置
#       ...
#       obj.init_var(xxx)                       # 状态变量和构件初始化
# 5. class构造函数输入仅包括最重要以及经常修改的参数，其他算法参数在set_param里面设置
####################

import time
import cv2
import numpy as np

EPS=np.float32(1e-16)   # 防止除零错
SPEED_OF_LIGHT=np.float32(299792458.0)

## 深度图预处理类
class depth_img_proc_c:
    
    ## 加载配置参数
    def set_param(self,img_wid,img_hgt):
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        self.IMG_SHAPE=(self.IMG_HGT,self.IMG_WID)
        
        # 距离切割标志和门限
        self.dist_mask=True 
        self.dist_dmin=np.float32(0.5)
        self.dist_dmax=np.float32(1.2)
        
        # 亮度切割标志和门限
        self.amp_mask =False
        self.amp_th   =np.float32(50.0)

        # 飞散点切割标志和门限
        self.fly_noise_filter=fly_noise_mask_c()
        self.var_mask =True
        self.var_th   =np.float32(0.01)

        # 空洞填充
        self.fill_hole=False
        
        # 空域中值滤波器
        self.median_filter    =False  
        self.MEDIAN_FILTER_WIN=3
        
        # 空域高斯（低通）滤波器
        self.gaussian_filter    =False  
        self.GAUSSIAN_FILTER_WIN=3
        self.GAUSSIAN_FILTER_S  =np.float32(0)

        # 时域1阶IIR滤波器
        self.iir_filter     =False  
        self.ciir_filter    =False  # (注意，仅用于TOF相机，并且需要知道深度值和相位的转换方法，见对应class的成员变量scale_d2p)
        self.IIR_FILTER_ALPHA=np.float32(0.5)   # 注意，过于接近1会导致拖尾

        # FIR滤波器
        self.wfir_filter    =False
        self.WFIR_SZ        =4
        
        # 复数FIR滤波器
        # (注意，仅用于TOF相机，并且需要知道深度值和相位的转换方法，见对应class的成员变量scale_d2p)
        self.cfir_filter    =False
        self.CFIR_SZ        =4
        
        # 时域中值滤波器
        self.median3t_filter=False
        
        # 空域导向滤波器
        # (注意：会导致距离失真)
        self.gif_filter     =False
        self.GIF_WIN        =16   # 导向滤波器平滑窗口尺寸（不能太小）
        return


    ## 初始化
    def __init__(self,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(img_wid,img_hgt) # 加载缺省配置参数
            self.init_var()                 # 初始化状态变量和构件
        return


    ## 初始化状态变量和构件
    def init_var(self):
        # 图像处理器对象
        self.iir  =img_iir_filter_c(self.IIR_FILTER_ALPHA,self.IMG_WID,self.IMG_HGT)
        self.ciir =img_cplx_iir_filter_c(self.IIR_FILTER_ALPHA,self.IMG_WID,self.IMG_HGT)
        self.wfir =amp_weighted_dep_filter_c(self.WFIR_SZ,self.IMG_WID,self.IMG_HGT)    # 光强加权FIR滤波器
        self.mid3t=mid3_time_filter_c(self.IMG_WID,self.IMG_HGT)                        # 时域中值滤波器
        self.cfir =amp_dep_cplx_filter_c(self.CFIR_SZ,self.IMG_WID,self.IMG_HGT)        # 复数FIR滤波器
        self.gif  =img_gif_c(self.GIF_WIN,self.GIF_WIN)                                 # 导向滤波器        
        return


    # 基于距离的像素过滤，计算屏蔽码mask
    def calc_dist_mask(self,img_dep):
        mask=np.full(self.IMG_SHAPE,True,dtype=bool)
        if self.dist_dmin>=0.0 and self.dist_dmax>=0.0 and self.dist_dmax<=self.dist_dmin: return mask
        if self.dist_dmin>=0.0: mask=np.bitwise_and(mask,(img_dep>=self.dist_dmin).astype(bool))
        if self.dist_dmax>=0.0: mask=np.bitwise_and(mask,(img_dep<=self.dist_dmax).astype(bool))
        return mask


    ## 计算基于强度的像素过滤
    def calc_amp_mask(self,img_amp): 
        return img_amp>=self.amp_th


    ## 像素过滤，计算用户选中的像素过滤器
    def calc_mask(self,img_dep, img_amp=None, mask=None):
        if mask is None:
            mask=np.full(self.IMG_SHAPE,True,dtype=bool)
        if self.dist_mask: mask=np.bitwise_and(mask,self.calc_dist_mask(img_dep))
        if self.amp_mask : 
            if img_amp is not None:
                mask=np.bitwise_and(mask,self.calc_amp_mask(img_amp))
        if self.var_mask : 
            mask=np.bitwise_and(mask,self.calc_var_mask(img_dep,self.var_th))
        return mask


    ## 空域飞散点过滤器
    def calc_var_mask(self,img_dep,th):
        return self.fly_noise_filter.calc(img_dep,th=th,win=self.GAUSSIAN_FILTER_WIN,mode='td')

    ## 图像滤波
    def calc_filter(self,img_dep,img_amp):
        img_out=img_dep.copy()
        
        # 时域滤波
        if self.iir_filter     : img_out=self.iir .calc(img_out)
        if self.ciir_filter    : img_out=self.ciir.calc(img_out,img_amp)
        if self.cfir_filter    : img_out=self.cfir.calc(img_out,img_amp)
        if self.wfir_filter    : img_out=self.wfir.calc(img_out,img_amp)
        if self.median3t_filter: img_out=self.mid3t.calc(img_out)
        
        # 空域中值
        if self.median_filter: 
            img_out=cv2.medianBlur(img_out.astype(np.float32),self.MEDIAN_FILTER_WIN)
            
        # 空域Gaussian低通滤波
        if self.gaussian_filter: 
            img_out=cv2.GaussianBlur(img_out,(self.GAUSSIAN_FILTER_WIN,self.GAUSSIAN_FILTER_WIN),self.GAUSSIAN_FILTER_S)

        # 空域导向滤波
        if self.gif_filter: 
            img_out=self.gif.calc(img_out,img_amp)

        return img_out


    ## 转换成uint8格式的数据，
    # 幅度范围调整到恰好在(dmin,dmax)之间
    def img_to_uint8(self,img):
        img_clamp=img-self.dist_dmin        
        if self.dist_dmax>self.dist_dmin: img_clamp=img_clamp*np.float32(1.0/(self.dist_dmax-self.dist_dmin))
        img_clamp[img_clamp<0.0]=np.float32(0.0)
        img_clamp[img_clamp>1.0]=np.float32(1.0)
        img_clamp*=np.float32(255.0)
        img_uint8=np.uint8(img_clamp)
        return img_uint8


    ## 图像预处理
    def calc(self,img_dep,img_amp=None,mask=None):
        img=self.calc_filter(img_dep,img_amp)       # 数据滤波
        mask=self.calc_mask(img_dep,img_amp,mask)   # 屏蔽码计算
        if self.fill_hole: img,mask=img_fill_hole(img,mask)
        img_u8=self.img_to_uint8(img)               # 格式转换
        return (img,img_u8,mask)


## 一阶IIR时域滤波器
class img_iir_filter_c:
    def set_param(self,alpha,img_wid,img_hgt):
        self.set_alpha(np.float32(alpha))
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        self.IMG_SHAPE=(self.IMG_HGT,self.IMG_WID)
        return


    def __init__(self,alpha=0.85,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(alpha,img_wid,img_hgt)   # 加载缺省配置参数
            self.init_var()                         # 初始化状态变量和构件
        return


    def init_var(self):
        self.imgf=np.zeros(self.IMG_SHAPE,dtype=np.float32)
        return


    def calc(self,img,mask=None):
        if mask is None:
            self.imgf=self.imgf*self.ALPHA+self.BETA*img
        else:
            self.imgf[mask]=self.imgf[mask]*self.ALPHA+self.BETA*img[mask]
        return self.imgf

    
    def set_alpha(self,alpha):
        self.ALPHA=np.float32(alpha)
        self.BETA =np.float32(1.0-self.ALPHA)
        return


## 一阶IIR复数时域滤波器
# 注意：成员变量scale_d2p用于将深度值转成相位，该算法仅仅对于TOF深度相机，这些相机输出的深度值和相位对应
class img_cplx_iir_filter_c:
    ## 设置配置参数
    def set_param(self,alpha,img_wid,img_hgt):
        self.set_alpha(alpha)
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        self.IMG_SHAPE=(self.IMG_HGT,self.IMG_WID)
        
        self.scale_d2p=np.float32(2.0*np.pi/(SPEED_OF_LIGHT/48.0e6/2.0))    # 将深度转成弧度
        self.scale_p2d=np.float32(1.0/(self.scale_d2p+EPS))                 # 将弧度转成深度
        return


    ## 初始化
    def __init__(self,alpha=0.85,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(alpha,img_wid,img_hgt)   # 加载缺省配置参数
            self.init_var()                         # 初始化状态变量和构件
        return
        
    
    # 初始化状态变量
    def init_var(self):
        self.sum_cos=np.zeros(self.IMG_SHAPE,dtype=np.float32)
        self.sum_sin=np.zeros(self.IMG_SHAPE,dtype=np.float32)
        return


    def calc(self,img_dep,img_amp):
        img_pha=img_dep*self.scale_d2p
        
        img_cos=np.float32(np.cos(img_pha))*img_amp
        img_sin=np.float32(np.sin(img_pha))*img_amp
        
        self.sum_cos=self.sum_cos*self.ALPHA+img_cos*self.BETA
        self.sum_sin=self.sum_sin*self.ALPHA+img_sin*self.BETA
        
        img_out = np.arctan2(self.sum_sin,self.sum_cos)*self.scale_p2d
        return img_out


    def set_alpha(self,alpha):
        self.ALPHA=np.float32(alpha)
        self.BETA =np.float32(1.0-self.ALPHA)
        return


## 光强度加权FIR滤波器
class amp_weighted_dep_filter_c:
    # 设置配置参数
    def set_param(self,buf_sz,img_wid,img_hgt):
        self.BUF_SZ=buf_sz
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        return


    # 初始化
    def __init__(self,buf_sz=10,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(buf_sz,img_wid,img_hgt)  # 加载缺省配置参数
            self.init_var()                         # 初始化状态变量和构件
        return


    # 初始化状态变量
    def init_var(self):
        self.buf_amp_dep=np.zeros((self.IMG_HGT,self.IMG_WID,self.BUF_SZ))
        self.buf_amp    =np.zeros((self.IMG_HGT,self.IMG_WID,self.BUF_SZ))
        
        self.sum_amp_dep=np.zeros((self.IMG_HGT,self.IMG_WID))
        self.sum_amp    =np.zeros((self.IMG_HGT,self.IMG_WID))
        
        self.idx=0
        return


    # 滤波
    def calc(self,img_dep,img_amp):
        amp_dep=img_dep*img_amp
        
        self.sum_amp_dep+=amp_dep-self.buf_amp_dep[:,:,self.idx]
        self.sum_amp    +=img_amp-self.buf_amp    [:,:,self.idx]
        
        self.buf_amp_dep[:,:,self.idx]=amp_dep
        self.buf_amp    [:,:,self.idx]=img_amp
        
        self.idx=(self.idx+1)%self.BUF_SZ
        
        return self.sum_amp_dep/(self.sum_amp+EPS)


## 复数FIR滤波器
# 注意：成员变量scale_d2p用于将深度值转成相位，该算法仅仅对于TOF深度相机，这些相机输出的深度值和相位对应
class amp_dep_cplx_filter_c:
    # 设置配置参数
    def set_param(self,buf_sz,img_wid,img_hgt):
        self.BUF_SZ=buf_sz
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt

        self.scale_d2p=2.0*np.pi/(SPEED_OF_LIGHT/48.0e6/2.0)    # 将深度转成弧度
        self.scale_p2d=1.0/(self.scale_d2p+EPS)  # 将弧度转成深度
        return


    # 初始化
    def __init__(self,buf_sz=10,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(buf_sz,img_wid,img_hgt)  # 加载缺省配置参数
            self.init_var()                         # 初始化状态变量和构件
        return


    # 初始化状态变量
    def init_var(self):
        self.buf_cos=np.zeros((self.IMG_HGT,self.IMG_WID,self.BUF_SZ))
        self.buf_sin=np.zeros((self.IMG_HGT,self.IMG_WID,self.BUF_SZ))

        self.sum_cos=np.zeros((self.IMG_HGT,self.IMG_WID))
        self.sum_sin=np.zeros((self.IMG_HGT,self.IMG_WID))
        
        self.idx=0
        return


    # 滤波
    def calc(self,img_dep,img_amp):
        img_pha=img_dep*self.scale_d2p
        
        img_cos=np.cos(img_pha)*img_amp
        img_sin=np.sin(img_pha)*img_amp
        
        self.sum_cos+=img_cos-self.buf_cos[:,:,self.idx]
        self.sum_sin+=img_sin-self.buf_sin[:,:,self.idx]
        
        self.buf_cos[:,:,self.idx]=img_cos
        self.buf_sin[:,:,self.idx]=img_sin

        self.idx=(self.idx+1)%self.BUF_SZ
        
        return np.arctan2(self.sum_sin,self.sum_cos)*self.scale_p2d


## 时域中值滤波器
class mid3_time_filter_c:
    # 设置配置参数
    def set_param(self,img_wid,img_hgt):
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        return


    # 初始化
    def __init__(self,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(img_wid,img_hgt) # 加载缺省配置参数
            self.init_var()                 # 初始化状态变量和构件
        return


    # 初始化状态变量
    def init_var(self):
        self.buf_dep=np.zeros((self.IMG_HGT,self.IMG_WID,3))
        self.idx=0
        return


    def calc(self,img_dep):
        self.buf_dep[:,:,self.idx]=img_dep
        img_sum=np.sum(self.buf_dep,axis=2)
        img_max=np.max(self.buf_dep,axis=2)
        img_min=np.min(self.buf_dep,axis=2)
        self.idx=(self.idx+1)%3        
        return img_sum-img_max-img_min


## 图像的矩形窗口平滑滤波器
#   使用了积分图算法，降低运算量
# 输入
#   img_in:     待平滑图像(2D)矩阵
#   wid,hgt:    平滑窗口尺寸
#   normalize:  True表示平滑需要输出除以平滑窗口面积
# 输出
#   img_out:    平滑后的图像(2D)矩阵
# TODO
#   可以用cv2.boxfilter取代
def rect_avg_filter(img_in,win_wid=3,win_hgt=3,normalize=True):
    img_hgt,img_wid=img_in.shape
    
    # 尺寸修正
    win_hgt+=1
    win_wid+=1
    
    # 求积分图
    img_sum=np.cumsum(img_in,axis=0)
    img_sum=np.cumsum(img_sum,axis=1)
    
    # ABCD是矩形窗的四个角
    A=img_sum.flatten()
    B=A[win_wid-1:]
    C=A[img_wid*(win_hgt-1):]
    D=C[win_wid-1:]
    
    # F是线条化的平滑输出
    N=len(D)
    F=(A[:N]+D[:N])-(B[:N]+C[:N])   
    if normalize: F*=1.0/float((win_wid-1)*(win_hgt-1))
    
    # 重构原图，保持位置对齐
    img_out0=np.hstack((np.zeros(img_wid*(win_hgt-1)+win_wid-1),F))
    img_out0.shape=img_hgt,img_wid
    
    hgt2,wid2=int(win_hgt/2),int(win_wid/2)
    img_out=np.zeros((img_hgt,img_wid))
    img_out[hgt2:img_hgt-(win_hgt-hgt2)+1,wid2:img_wid-(win_wid-wid2)+1]=img_out0[win_hgt-1:,win_wid-1:]
    img_out.shape=img_in.shape
    return img_out


## 导向滤波运算
class img_gif_c:
    def set_param(self,win_wid,win_hgt):
        self.win_wid=win_wid
        self.win_hgt=win_hgt
        return

        
    ## 初始化
    # 输入
    #   win_wid,win_hgt：平滑窗口尺寸
    def __init__(self,win_wid=8,win_hgt=8,init=True):
        if init:
            self.set_param(win_wid,win_hgt)
            self.init_var()
        return

    ## 状态变量和构建初始化    
    def init_var(self):
        return


    ## 滤波计算
    # 输入
    #   img_P:      待滤波图(2D)矩阵
    #   img_I:      参考图(2D)矩阵
    #   output_all: 返回模式，等于True表示返回包括了中间数据：mean_I, mean_P, mean_a, mean_b
    # 输出
    #   img_Q:  滤波输出
    def calc(self,img_P, img_I, output_all=False):
        mean_I =rect_avg_filter(img_I,win_wid=self.win_wid,win_hgt=self.win_hgt)
        mean_P =rect_avg_filter(img_P,win_wid=self.win_wid,win_hgt=self.win_hgt)
        
        corr_I =rect_avg_filter(img_I*img_I,win_wid=self.win_wid,win_hgt=self.win_hgt)
        corr_IP=rect_avg_filter(img_I*img_P,win_wid=self.win_wid,win_hgt=self.win_hgt)
        
        var_I  =corr_I -mean_I*mean_I
        cov_IP =corr_IP-mean_I*mean_P
        
        a=cov_IP/(var_I+EPS)
        b=mean_P-a*mean_I
        
        mean_a=rect_avg_filter(a,win_wid=self.win_wid,win_hgt=self.win_hgt)
        mean_b=rect_avg_filter(b,win_wid=self.win_wid,win_hgt=self.win_hgt)
        
        img_Q=mean_a*img_I+mean_b
        return (img_Q, mean_I, mean_P, mean_a, mean_b) if output_all else img_Q 


## 填充空洞
# 输入
#   img:    待处理图(2D)矩阵
#   mask:   图像屏蔽码
#   win:    填充滤波器窗口尺寸
#   th:     填充像素要求窗口内有效元素占的数量(0~win**2)
# 输出
#   img_out:填充后的图(2D)矩阵
#   mask_out:填充后图像的屏蔽码
# TODO
#   可以用cv2.boxfilter快速查找可以填补的空洞
def img_fill_hole(img,mask,win=3,th=4):
    # 统计每个像素的附近有效像素数量
    mask_sum=rect_avg_filter(mask.astype(np.int),win_wid=win,win_hgt=win,normalize=False)
    mask_fill=(mask_sum>=th)    # 根据邻近像素数量，确定是否填充
    
    # 标示出空洞位置
    mask_hole=np.bitwise_and(~mask,mask_fill)

    # 填充空洞
    img_out=img.copy()
    img_out[~mask]=0    # 原图无效区域置零
    img_sum=rect_avg_filter(img_out,win_wid=win,win_hgt=win,normalize=False)    # 计算滑动窗口内的像素和
    img_out[mask_hole]=img_sum[mask_hole]/mask_sum[mask_hole]   # 对于空洞位置的像素，赋值为滑动窗口内的像素平均值
    
    # 更新填补了空洞后的mask
    mask_out=np.bitwise_or(mask,mask_hole)
    
    return img_out,mask_out

## 用特定的核函数对图像滤波
# 输入
#   img_dep:待处理图深度图
#   mask:   图像屏蔽码
#   it:     重复滤波次数
#   ker:    滤波核
# 输出
#   img_out:滤波输出图
# 注意：
#   会根据mask补偿像素空空洞对应的滤波结果
def img_noise_filter(img,mask=None,it=1,ker=None):
    if ker is None:    # 缺省使用拉普拉斯滤波核
        ker = np.array([[1,2,1],
                        [2,0,2],
                        [1,2,1]])/12.0
    
    # 要求重复滤波多次?
    if it>1:
        img_out=img_noise_filter(img,mask,it=1,ker=ker)
        for _ in range(1,it):
            img_out=img_noise_filter(img_out,mask,it=1,ker=ker)
        return img_out
    
    # 计算滤波结果    
    img_out=cv2.filter2D(img,-1,ker)
    
    # 根据mask补偿像素空洞对应的滤波结果
    if mask is not None:    
        pix_cnt=cv2.filter2D(mask.astype(np.float),-1,ker)
        pix_sel=(pix_cnt>0)
        img_out[pix_sel]=img_out[pix_sel]/pix_cnt[pix_sel]
        
    return img_out
        
    
## 计算图像的时域方差
class img_var_calc_c:
    def set_param(self,buf_sz,img_wid,img_hgt):
        self.BUF_SZ=buf_sz
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        return
        
        
    def __init__(self,buf_sz=10,img_wid=256,img_hgt=256,init=True):
        if init:
            self.set_param(buf_sz,img_wid,img_hgt)
            self.init_var()
        return
            
    def init_var(self):
        self.pos=0
        self.img_buf=np.zeros((self.IMG_HGT,self.IMG_WID,self.BUF_SZ))
            
        self.img_sum    =np.zeros((self.IMG_HGT,self.IMG_WID))
        self.img_sqr_sum=np.zeros((self.IMG_HGT,self.IMG_WID))
        self.img_mean   =np.zeros((self.IMG_HGT,self.IMG_WID))
        self.img_var    =np.zeros((self.IMG_HGT,self.IMG_WID))

        self.valid_cnt=0
        return


    # 检测缓冲区是否被充满（方差和均值数据是否有效）    
    def valid(self): return self.valid_cnt>=self.BUF_SZ


    def calc(self,img):
        if self.valid_cnt<self.BUF_SZ:
            self.valid_cnt+=1
        
        self.img_sum+=img-self.img_buf[:,:,self.pos]
        self.img_mean=self.img_sum/float(self.BUF_SZ)
        self.img_sqr_sum+=img**2-self.img_buf[:,:,self.pos]**2
        
        self.img_var=np.sqrt(np.abs(self.img_sqr_sum/float(self.BUF_SZ)-self.img_mean**2))
        
        self.img_buf[:,:,self.pos]=img
        self.pos=(self.pos+1)%self.BUF_SZ

        return self.img_mean, self.img_var


## 飞散噪声过滤
# 实现3中过滤模式：
#   'td' ：通过连续三帧深度图的像素移动距离范围确定飞散点
#   'bxf'：通过比较平滑滤波结果和当前像素的距离差确定飞散点
#   'mdf'：通过比较中值滤波结果和当前像素的距离差确定飞散点    
class fly_noise_mask_c:
    ## 初始化
    # 输入
    #    mode   过滤模式，
    #           'td' ：通过连续三帧深度图的像素移动距离范围确定飞散点
    #           'bxf'：通过比较平滑滤波结果和当前像素的距离差确定飞散点
    #           'mdf'：通过比较中值滤波结果和当前像素的距离差确定飞散点
    def __init__(self,mode='td'):
        self.frame=[None,None,None]
        self.idx=0
        self.init=True
        self.mode=mode
    
    ## 飞散点过滤
    # 输入
    #   img_dep 待检测深度图
    #   th      飞散点判定门限，单位是距离，越小越容易判为飞散点
    #   win     飞散点检测窗口，用于bxf和mdf模式
    #   mode    过滤模式（如果不提供，缺省取类初始化的值）
    # 输出
    #   图像mask，其中取值False的元素对应滤除的飞散点
    def calc(self,img_dep,th=0.02,win=3,mode=None):
        if mode is None: mode=self.mode
        
        if   mode =='td': return self.calc_td(img_dep,th)
        elif mode=='bxf': return self.calc_bxf(img_dep,th,win)
        elif mode=='mdf': return self.calc_mdf(img_dep,th,win)
        return
                
    ## 基于平滑滤波的飞散点检测
    # 输入
    #   img_dep 待检测深度图
    #   th      飞散点判定门限，单位是距离，越小越容易判为飞散点
    #   win     飞散点检测窗口，用于bxf和mdf模式
    # 输出
    #   图像mask，其中取值False的元素对应滤除的飞散点
    def calc_bxf(self,img_dep,th=0.02,win=3):
        img_blur=cv2.boxFilter(img_dep,-1,(win,win))  
        return np.abs(img_dep-img_blur)<th
        
    ## 基于中值滤波的飞散点检测
    # 输入
    #   img_dep 待检测深度图
    #   th      飞散点判定门限，单位是距离，越小越容易判为飞散点
    #   win     飞散点检测窗口，用于bxf和mdf模式
    # 输出
    #   图像mask，其中取值False的元素对应滤除的飞散点
    def calc_mdf(self,img_dep,th=0.02,win=3):
        img_blur=cv2.medianBlur(img_dep,win)  
        return np.abs(img_dep-img_blur)<th
        
    ## 基于时间域连续三帧深度图像素深度变化范围的飞散点检测
    # 输入
    #   img_dep 待检测深度图
    #   th      飞散点判定门限，单位是距离，越小越容易判为飞散点
    # 输出
    #   图像mask，其中取值False的元素对应滤除的飞散点
    def calc_td(self,img_dep,th=0.02):
        # 保存图片
        self.frame[self.idx]=img_dep.copy()
        self.idx+=1
        self.idx%=len(self.frame)
        
        if self.init:
            self.init=False if (self.idx==0) else True
            return np.ones_like(img_dep,dtype=bool)
        
        # 计算3帧图像时域波动
        frame_max=np.maximum(np.maximum(self.frame[0],self.frame[1]),self.frame[2])
        frame_min=np.minimum(np.minimum(self.frame[0],self.frame[1]),self.frame[2])
        
        return frame_max-frame_min<th


## 基于深度图时域改变量的自适应背景检测
# 1. 计算深度图的均值和方差
# 2. 找出静止区域，即方差小于门限的区域
# 3. 根据静止区域更新背景数据，优先考虑远距离的数据
# TODO
# 图像中阴影和远距离噪声无法被记录为成背景
# 图像中固定的前景遮挡区域无法识别为前景（该区域无法得到背景更新数据）
# 1. 增加色彩记录，根据色彩稳定度确定前后景
# 2. 增加灰度分析，根据灰度稳定度确定前后景
# 3. 增加距离连通域检测，确定前景内点，去除当前点的“假”背景
# 4. 增加色彩和物体连通域检测
class var_bg_det_c:
    
    ## 配置参数初始化
    def set_param(self,img_wid,img_hgt,img_bg_ref=None,img_bg_th=None):
        self.IMG_WID=img_wid
        self.IMG_HGT=img_hgt
        self.IMG_SHAPE=(self.IMG_HGT,self.IMG_WID)

        # 可配置参数
        self.ALPHA=0.99                 # 背景遗忘因子(0.99对应半衰期69，alpha和半衰期K关系是：alpha=0.5**(1/K))
        self.BETA=1.0-self.ALPHA        # 背景更新因子
    
        self.QUIET_TH_VAR=5.0e-3        # 图像静止检测的方差门限
        self.QUIET_TH_MEAN=5.0e-3       # 图像静止检测的均值比较门限
        
        self.ENABLE_BG_NEAR_CLIP=True   # 背景前移约束
        self.NEAR_TH=5.0e-3             # 背景更新允许的移近量门限
            
        self.BG_TH=10.0e-3              # 区分前/背景的距离门限
            
        self.ENABLE_FAR_BG_DET=True     # 是否将深度图后移判为背景
        self.FAR_BG_TH=10.0e-3          # 优先判别为背景的深度图后移量门限
            
        self.ENABLE_NEAR_FG_DET=True    # 是否将深度图前移判为情景
        self.NEAR_FG_TH=10.0e-3         # 优先判别为前景的深度图前移量门限
        
        self.ENABLE_DEP_CHG_DET=False   # 是否使用相邻两帧的深度图差别辨别情景
        self.DEP_CHG_TH=10.0e-3         # 相邻帧移动检测门限, 超过门限被优先判为前景
        
        self.IMG_VAR_DET_SZ=12          # 图像运动检测器记录的图像帧长度
    
        self.DENOISE=False              # 形态学去噪参数
        self.DENOISE_KER_SZ=2
        self.DENOISE_KER=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.DENOISE_KER_SZ,self.DENOISE_KER_SZ))  # 去噪核(会被多次使用）
        
        self.ENABLE_IMG_BG_REF=True if img_bg_ref is not None else False
        self.img_bg_ref=img_bg_ref      # 预定义的参考背景图
        
        self.ENABLE_IMG_BG_TH=True if img_bg_th is not None else False
        self.img_bg_th =img_bg_th       # 预定义的参考背景切割门限
        
        return


    def __init__(self,img_wid=256,img_hgt=256,img_bg_ref=None, img_bg_th=None, init=True):
        if init: 
            self.set_param(img_wid,img_hgt,img_bg_ref,img_bg_th) # 初始化参数
            self.init_var()                 # 初始化状态变量和构件
        return


    ## 初始化状态数据
    def init_var(self):
        self.img_bg  =np.zeros(self.IMG_SHAPE)                  # 存放背景数据
        self.img_var_det=img_var_calc_c(buf_sz=self.IMG_VAR_DET_SZ,
                                        img_wid=self.IMG_WID,
                                        img_hgt=self.IMG_HGT)   # 图像变化检测器
        self.img_dep_last=None                                  # 最近的深度图
        
        self.bg_quiet=np.zeros(self.IMG_SHAPE)  # 存放“安静”背景
        return

    
    ## 输入img_dep像素单位是M
    # 输入
    #   img_dep：深度图，2D数组
    # 输出
    #   img_bg：背景深度图
    #   mask：背景屏蔽码（2D bool数组，元素=True表明是背景）
    def calc(self,img_dep):
        # 图像运动检测
        img_mean,img_var=self.img_var_det.calc(img_dep)

        # 根据方差计算探测静止区域
        #if not self.img_var_det.valid(): return self.img_bg,np.zeros_like(img_dep,dtype=bool)
        quiet_mask=np.bitwise_and(img_var<self.QUIET_TH_VAR,                    # 方差小于门限
                                  np.abs(img_mean-img_dep)<self.QUIET_TH_MEAN)  # 当前像素没有运动

        # 优先接受更远的区域
        far_mask=np.bitwise_and(quiet_mask, img_dep>self.img_bg)
        if self.DENOISE:
            far_mask=cv2.morphologyEx(far_mask.astype(np.uint8),cv2.MORPH_OPEN ,self.DENOISE_KER)  # 开运算，去除杂散噪声
        self.img_bg[far_mask]=img_dep[far_mask]  

        # 提取静止区域的背景
        img_bg_quiet=self.img_bg[quiet_mask]
        
        # 计算静止区域内背景距离修正后结果
        img_bg_chg=img_bg_quiet*self.ALPHA+img_dep[quiet_mask]*self.BETA
        
        # 限制新的背景移近的速度
        if self.ENABLE_BG_NEAR_CLIP:
            near_mask=img_bg_quiet-img_bg_chg>self.NEAR_TH
            img_bg_chg[near_mask]=img_bg_quiet[near_mask]-self.NEAR_TH
        
        # 更新背景数据
        self.img_bg[quiet_mask]=img_bg_chg
        
        # 前/背景区分
        mask=self.bg_det(img_dep)    
        self.img_dep_last=img_dep
        return self.img_bg.copy(),mask


    ## 前/背景区分
    def bg_det(self,img_dep):
        # 当前图片背景属于背景的标识
        mask=np.abs(img_dep-self.img_bg)<self.BG_TH # 简单深度值和背景没有差异，认为是背景
        
        # 当前像素更远，优先认为是背景
        if self.ENABLE_FAR_BG_DET:
            mask=np.bitwise_or(mask,img_dep-self.img_bg>self.FAR_BG_TH)     
        
        # 当前像素更近，优先从背景扣除
        if self.ENABLE_NEAR_FG_DET:
            mask=np.bitwise_and(mask,~(img_dep+self.NEAR_FG_TH<self.img_bg))
        
        # 相邻帧出现运动的像素，先认为是背景
        if self.ENABLE_DEP_CHG_DET:
            if self.img_dep_last is not None:
                mask=np.bitwise_and(mask,~(np.abs(self.img_dep_last-img_dep)>self.DEP_CHG_TH))   
        
        # 靠近预定义的背景图的是后景
        if self.ENABLE_IMG_BG_REF:
            mask=np.bitwise_or(mask,np.abs(img_dep-self.img_bg_ref)<self.BG_TH)
        
        # 超过预定义的背景门限图的是后景
        if self.ENABLE_IMG_BG_TH:
            mask=np.bitwise_or(mask,img_dep>self.img_bg_th)

        return mask


## 将灰度图转成伪彩色图
def img_to_cmap(img,mask=None,vmin=-1,vmax=-1,color=cv2.COLORMAP_RAINBOW):
    if vmin<0: vmin=np.min(img)
    if vmax<0: vmax=np.max(img)
    if vmax==vmin: vmax=vmin+EPS
    
    img_norm=np.clip(np.float32(img-vmin)/np.float32(vmax-vmin),0.0,1.0)
    img_u8=np.uint8(img_norm*255)
    img_rgb=cv2.applyColorMap(255-img_u8,color)
    if mask is not None: img_rgb[~mask,:]=0
    return img_rgb

## 记录背景并保存
# 输入：
#   cam         相机设备
#   viewer      GUI
#   fname_dep   深度图存盘文件名
#   fname_amp   强度图存盘文件名
#   cum_cnt     计算背景的图像帧数量
# 输出：
#   背景数据
def calc_bg(cam,viewer=None,fname_dep=None,fname_amp=None,cum_cnt=100,skip=0):
    dep_frames=[]
    amp_frames=[]
    
    frame_id=cnt=0
    while cnt<cum_cnt+skip:
        # 获取数据
        img_dep,img_amp,frame_id_new=cam.get_dep_amp()
        if frame_id==frame_id_new:
            time.sleep(0.001)
            continue
        else:
            frame_id=frame_id_new
            cnt+=1
            if cnt%10==0: print('[%d]'%cnt) 
        if cnt<skip: continue
        
        # 保存数据
        dep_frames.append(img_dep.copy().flatten()) 
        amp_frames.append(img_amp.copy().flatten()) 

        # 显示图像
        if viewer is not None:
            viewer.update_pan_img_rgb(img_to_cmap(img_dep,mask=None,vmin=0,vmax=2),pan_id=(0,0))
            if img_amp is not None: viewer.update_pan_img_gray(img_amp*0.5,pan_id=(0,1))
        
            # 屏幕刷新
            viewer.update()
            evt,_=viewer.poll_evt()
            if evt is not None: 
                if evt=='quit': break 
    
    # 计算背景
    bg_dep=dep_frames[0]
    for n in range(1,cum_cnt): bg_dep+=dep_frames[n]
    bg_dep/=float(cum_cnt)
    
    bg_amp=amp_frames[0]
    for n in range(1,cum_cnt): bg_amp+=amp_frames[n]
    bg_amp/=float(cum_cnt)
    
    # 保存背景
    if fname_dep is not None: bg_dep.astype(np.float32).tofile(fname_dep)   
    if fname_amp is not None: bg_dep.astype(np.float32).tofile(fname_amp)   
    return bg_dep, bg_amp


## 读入将静态场景深度图文件，求出平均保存为背景文件
# 输入文件格式为：320x240的float32深度数据
def calc_bg_from_file(fname_in,img_sz=76800,cum_cnt=np.inf):
    fd=open(fname_in,'rb')
    bg=np.zeros(img_sz,dtype=np.float32)
    cnt=0
    while cnt<cum_cnt:
        try: 
            d=np.fromfile(fd,dtype=np.float32,count=img_sz)
            if len(d)<img_sz: break
            bg+=d
            cnt+=1
        except:
            break
    fd.close()
    bg/=np.float32(cnt)
    return bg 


## 基于背景深度数据，检测删除了背景的mask
def gen_fg_mask(bg,img_dep,th=0.1,mask_in=None):
    return (bg-img_dep<th) if mask_in is None else np.bitwise_and(mask_in,bg-img_dep<th) 



