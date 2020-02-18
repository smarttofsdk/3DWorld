#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

####################
#    深度图处理库    #
####################
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
    def calc_amp_mask(self,img_amp): return img_amp>=self.amp_th


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


    ## 飞散点过滤器
    def calc_var_mask(self,img_dep,th):
        img_blur=cv2.GaussianBlur(img_dep,(self.GAUSSIAN_FILTER_WIN,self.GAUSSIAN_FILTER_WIN),self.GAUSSIAN_FILTER_S)
        return np.abs(img_dep-img_blur)<th


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
            img_out=cv2.medianBlur(img_out,self.MEDIAN_FILTER_WIN)
            
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


    def calc(self,img):
        self.imgf=self.imgf*self.ALPHA+self.BETA*img
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
#   可以用形态学滤波查找空洞
def img_fill_hole(img,mask,win=3,th=4):
    # 统计每个附近的有效像素数量
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
    def set_param(self,img_wid,img_hgt):
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
        
        return


    def __init__(self,img_wid=256,img_hgt=256,init=True):
        if init: 
            self.set_param(img_wid,img_hgt) # 初始化参数
            self.init_var()                 # 初始化状态变量和构件
        return


    ## 初始化状态数据
    def init_var(self):
        self.img_bg  =np.zeros(self.IMG_SHAPE)                  # 存放背景数据
        self.img_var_det=img_var_calc_c(buf_sz=self.IMG_VAR_DET_SZ,
                                        img_wid=self.IMG_WID,
                                        img_hgt=self.IMG_HGT)   # 图像变化检测器
        self.img_dep_last=None                                  # 最近的深度图
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
        if not self.img_var_det.valid(): return self.img_bg,np.zeros_like(img_dep,dtype=bool)
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
        if self.ENABLE_FAR_BG_DET:
            mask=np.bitwise_or(mask,img_dep-self.img_bg>self.FAR_BG_TH)     # 当前像素更远，优先认为是背景
        
        if self.ENABLE_NEAR_FG_DET:
            mask=np.bitwise_and(mask,~(img_dep+self.NEAR_FG_TH<self.img_bg))# 当前像素更近，优先从背景扣除
        
        if self.ENABLE_DEP_CHG_DET:
            if self.img_dep_last is not None:
                mask=np.bitwise_and(mask,~(np.abs(self.img_dep_last-img_dep)>self.DEP_CHG_TH))   # 相邻帧出现运动的像素，从背景扣除
        return mask


## 将灰度图转成伪彩色图
def img_to_cmap(img,mask=None,vmin=-1,vmax=-1,color=cv2.COLORMAP_RAINBOW):
    if vmin<0: vmin=np.min(img)
    if vmax<0: vmax=np.max(img)


    img_norm=np.clip(np.float32(img-vmin)/np.float32(vmax-vmin),0.0,1.0)
    img_u8=np.uint8(img_norm*255)
    img_rgb=cv2.applyColorMap(255-img_u8,color)
    if mask is not None: img_rgb[~mask,:]=0
    return img_rgb


####################
# 测试代码
####################
if __name__ == '__main__':
    import time,sys,os
    
    # 装载私有库
    sys.path.append('./')
    from global_cfg import *
    from pyg_viewer import *
    from depth_cam_tools import *

    TEST_3D_VIEW=True       # 是否显示3D点云
    TEST_BG_DET=True        # 是否检测背景并扣除

    ####################
    # 点云变换参数
    ####################
    ax=ay=0.0                       # 点云旋转角度
    cz=1.0                          # 点云旋转中心
    mz=-0                           # 点云观察点位置
    T=np.eye(4,dtype=np.float32)    # 点云变换矩阵
    
    dmin=0.0    # 点云距离过滤门限（距离范围外的点被清除）
    dmax=2.0
    
    ####################
    # 深度图预处理工具以及配置
    ####################
    #dep_proc=depth_img_proc_c(img_wid=512,img_hgt=424)
    dep_proc=depth_img_proc_c(init=False)
    dep_proc.set_param(img_wid=512,img_hgt=424)
    
    dep_proc.dist_dmin=0.5
    dep_proc.dist_dmax=1.2
    
    dep_proc.dist_mask      =True   # 距离切割标志和门限
    dep_proc.amp_mask       =False  # 亮度切割标志和门限
    dep_proc.var_mask       =True   # 飞散点切割标志和门限
    
    dep_proc.fill_hole      =False  # 空洞填充
    
    dep_proc.median_filter  =False  # 空域中值滤波器  
    dep_proc.gaussian_filter=False  # 空域高斯（低通）滤波器  
    dep_proc.iir_filter     =False  # 时域1阶IIR滤波器  
    dep_proc.wfir_filter    =False  # FIR滤波器
    dep_proc.cfir_filter    =False  # 复数FIR滤波器(注意，仅用于TOF相机，并且需要知道深度值和相位的转换方法，见对应class的成员变量scale_d2p)
    dep_proc.median3t_filter=False  # 时域中值滤波器
    dep_proc.gif_filter     =False  # 空域导向滤波器(注意：会导致距离失真)
    
    dep_proc.init_var()
    
    ####################
    # 前背景处理工具以及配置
    ####################
    bg_det=var_bg_det_c(init=False)
    bg_det.set_param(img_wid=512,img_hgt=424)
    
    bg_det.ALPHA=0.99                 # 背景遗忘因子(0.99对应半衰期69，alpha和半衰期K关系是：alpha=0.5**(1/K))
    bg_det.BETA=1.0-bg_det.ALPHA      # 背景更新因子
    
    bg_det.QUIET_TH_VAR=5.0e-3        # 图像静止检测的方差门限
    bg_det.QUIET_TH_MEAN=5.0e-3       # 图像静止检测的均值比较门限
    
    bg_det.ENABLE_BG_NEAR_CLIP=True   # 背景前移约束
    bg_det.NEAR_TH=5.0e-3             # 背景更新允许的移近量门限
        
    bg_det.BG_TH=10.0e-3              # 区分前/背景的距离门限
        
    bg_det.ENABLE_FAR_BG_DET=True     # 是否将深度图后移判为背景
    bg_det.FAR_BG_TH=10.0e-3          # 优先判别为背景的深度图后移量门限
        
    bg_det.ENABLE_NEAR_FG_DET=True    # 是否将深度图前移判为情景
    bg_det.NEAR_FG_TH=10.0e-3         # 优先判别为前景的深度图前移量门限
    
    bg_det.ENABLE_DEP_CHG_DET=False   # 是否使用相邻两帧的深度图差别辨别情景
    bg_det.DEP_CHG_TH=10.0e-3         # 相邻帧移动检测门限, 超过门限被优先判为前景
    
    bg_det.IMG_VAR_DET_SZ=12          # 图像运动检测器记录的图像帧长度

    bg_det.init_var()

    ####################
    # 深度图处理工具
    ####################
    dep_trans=depth_cam_trans_c(img_wid=512,img_hgt=424)
    dep_trans.cam_param_f(3.727806701424671e2)    
    
    ####################
    # GUI显示器
    ####################
    viewer=pyg_viewer_c(pan=(1,3))
    
    ####################
    # 连续回放显示
    ####################
    
    # 鼠标动作跟踪
    mouse_down=False
    mouse_x=mouse_y=0
    
    # 回放文件
    fp_dep=open('../kinect/data/halfbody/data0_dep.bin','rb')
    fp_amp=open('../kinect/data/halfbody/data0_ir.bin','rb')
    
    frame_cnt=0
    while True:
        # 从数据中读取一帧深度图
        frame_dep=np.fromfile(fp_dep,dtype=np.int16,count=FRAME_DEP_SZ)
        if len(frame_dep)<FRAME_DEP_SZ:
            fp_dep.seek(0, os.SEEK_SET)
            frame_dep=np.fromfile(fp_dep,dtype=np.int16,count=FRAME_DEP_SZ)
            frame_cnt+=1
    
        # 从数据中读取一帧强度图
        frame_amp=np.fromfile(fp_amp,dtype=np.int16,count=FRAME_DEP_SZ)
        if len(frame_amp)<FRAME_DEP_SZ:
            fp_amp.seek(0, os.SEEK_SET)
            frame_amp=np.fromfile(fp_amp,dtype=np.int16,count=FRAME_DEP_SZ)
        
        # 单位转换
        img_dep=frame_dep.copy().astype(np.float32).reshape(424,512)/1000.0  # 注意，单位是M
        img_amp=frame_amp.copy().astype(np.float32).reshape(424,512)/32768.0*255.0
        
        mask=np.ones_like(img_dep,dtype=bool)
        
        # 背景检测
        if TEST_BG_DET:
            img_bg,mask_bg=bg_det.calc(img_dep)
            mask=np.bitwise_and(mask,~mask_bg)
            
        # 图像预处理
        img_dep_ori=img_dep.copy()
        img_dep,img_dep_u8,mask=dep_proc.calc(img_dep,img_amp,mask)
        
        # 显示处理前后图像差别
        img_rgb=img_to_cmap(img_dep_ori,mask=None,vmin=dmin,vmax=dmax)
        viewer.update_pan_img_rgb(img_rgb,pan_id=(0,1))
        img_rgb=img_to_cmap(img_dep,mask,vmin=dmin,vmax=dmax)
        viewer.update_pan_img_rgb(img_rgb,pan_id=(0,2))
        
        if TEST_3D_VIEW:
            # 将深度图转换成点云
            pc=dep_trans.depth_to_pcloud(img_dep,mask)
    
            #############################################
            # 
            # 在此加入额外的点云数据处理
            # 点云存放在数组Nx3的数组pc中，pc的每一行对应一个空间的点，pc的3列分别是点的x/y/z坐标
            # 
            #############################################
            
            # 点云变换，并将变换后的点云映射回深度图
            pc_new=pc_trans(T,pc)
            img_dep_new,mask=dep_trans.pcloud_to_depth(pc_new)
            
            # 将深度图转换成伪彩色，并更新对应的显示版面
            img_rgb=img_to_cmap(img_dep_new,mask,dmin,dmax)
            viewer.update_pan_img_rgb(img_rgb)
        
        # 刷新屏幕显示
        viewer.update()
        
        # 检查鼠标动作
        update_trans_mat=False
        evt,param=viewer.poll_evt()
        if evt is not None: 
            #print(evt)
            if evt=='md0': 
                mouse_down=True
                mouse_x,mouse_y=param[0],param[1]
            elif evt=='mu0': 
                mouse_down=False
            elif evt=='mm':
                if mouse_down:
                    dx=param[0]-mouse_x
                    dy=param[1]-mouse_y
                    mouse_x,mouse_y=param[0],param[1]
                    ax+=dy/50.0
                    ay-=dx/50.0
                    update_trans_mat=True
            elif evt=='mw0':
                mz+=0.1
                update_trans_mat=True
            elif evt=='mw1':
                mz-=0.1
                update_trans_mat=True
        
        # 根据鼠标动作更新显示的3D内容
        if update_trans_mat:
            T=pc_movz(-cz)
            T=np.dot(T,pc_rotx(ax))
            T=np.dot(T,pc_roty(ay))
            T=np.dot(T,pc_movz(cz))
            T=np.dot(T,pc_movz(mz))
    
    fp_dep.close()
    fp_amp.close()


