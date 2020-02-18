#!/usr/bin/python3
# coding=utf-8

import numpy as np
import math

DEBUG_PRINT=False

# 仿真模式
ENABLE_SIM=False
#ENABLE_SIM=True

USE_ZMQ=False
ZMQ_ADDR='tcp://192.168.7.2:56789'

#DEP_CAM_SCALE=0.5/250.0                             # 将原始dep数据转成M单位数据的比率因子
DEP_CAM_SCALE=0.5/140.0                             # 将原始dep数据转成M单位数据的比率因子


ENABLE_PC_ROTATE=True#False          # 需要点云旋转
PC_ROT_CZ=0.4                   # 点云旋转中心
PC_ROT_AX=math.pi/180.0*90.0    # 点云旋转角度(沿X轴)

####################
# 回放参数
####################
PLAYBACK_FNAME    ='D:/TOF_data/epc660/data/20171030'

PLAYBACK_FPS=7                  # 回放帧率
PLAYBACK_SKIP=0#620               # 回放跳过的帧
PLAYBACK_FRAME_CNT_MAX=np.inf   # 回放帧号上限

##################
# 图像预处理参数
##################
IMG_PROC_DMIN=0.0
IMG_PROC_DMAX=0.18

##################
# 深度相机配置
##################

import sys
sys.path.append('./')

# 数据帧尺寸
FRAME_WID       =320
FRAME_HGT       =240
FRAME_SZ        =FRAME_WID*FRAME_HGT
FRAME_BYTE_SZ   =FRAME_SZ*2

# 图像尺寸
IMG_WID=FRAME_WID
IMG_HGT=FRAME_HGT
IMG_SZ =IMG_WID*IMG_HGT

# 深度相机工具参数
DEP_CAM_ANGLE=62.0*(math.pi/180.0)
DEP_CAM_F=320.0/(2.0*math.tan(DEP_CAM_ANGLE/2.0))   # 像素坐标和物理尺寸的转换因子
DEP_CAM_EPS=1.0e-16
#DEP_CAM_SCALE=0.5/250.0                             # 将原始dep数据转成M单位数据的比率因子
DEP_CAM_FPS=7                                       # 相机帧率

##################
# 指尖跟踪算法参数
##################
FTIP_TRK_AMP_TH     =-1#250                 # 像素亮度(切割)门限
FTIP_TRK_DMIN       =-1#0.0                 # 像素距离(切割)门限
FTIP_TRK_DMAX       =-1#0.6                 # 像素距离(切割)门限

FTIP_TRK_LIST_SZ    =int(2*DEP_CAM_FPS)     # 记忆的图像帧数量
FTIP_TRK_MID_FT     =5                      # 中值滤波参数
FTIP_TRK_DIST_SEL   =50                     # 运算选择最靠近镜头的像素数量
FTIP_TRK_EPS        =1.0e-12                # 避免运算除零错的“松弛”因子
FTIP_TRK_VALID_TH   =0.9*FTIP_TRK_LIST_SZ   # 识别需要的有效帧（检测到指尖的）数量门限
FTIP_TRK_Q0_TH      =0.8*FTIP_TRK_DIST_SEL  # 判定指尖技术那结果是否有效的质量因子门限（用于确定有效之间像素数量是否足够）
FTIP_TRK_Q1_TH      =(IMG_WID+IMG_HGT)*0.1  # 判定指尖技术那结果是否有效的质量因子门限（用于确定指尖像素的分布范围是否足够小）
FTIP_TRK_HLINE_TH   = 2.0                   # 判定水平线门限
FTIP_TRK_VLINE_TH   = 2.0                   # 判定垂直线门限
FTIP_TRK_LINE45_TH  = 2.5                   # 判定水平线门限
FTIP_TRK_LINE135_TH = 2.5                   # 判定垂直线门限
FTIP_TRK_POINT_TH   = 20                    # 判定点门限
FTIP_TRK_CIRCLE_TH  = 2                     # 判定圆门限
FTIP_TRK_CIRCLE_ERR = 1.0-0.4**2            # 圆上点距离圆心相对距离最小值门限

FTIP_TRK_DENOISE_TH = 5                     # 去噪声算法门限（形态学滤波模板尺寸）
                
FTIP_TRK_BEND_TH    = -0.5                  # 指尖曲率判定门限（-1~+1,越大，越难检出指尖）
FTIP_TRK_BEND_TH2   = -0.0                  # 指尖曲率二次筛选判定门限
                
FTIP_TRK_MARK_SZ    = 4                     # 手指跟踪时显示的标记尺寸
                
FTIP_TRK_CT_BLK_SZ  = 2                     # 基于消块法检测轮廓序列算法的外块尺寸
FTIP_TRK_CT_BLK_SZ1 = FTIP_TRK_CT_BLK_SZ-1  # 基于消块法检测轮廓序列算法的内块尺寸

FTIP_TRK_CT_DOWN_SAMPLE = 2                 # 轮廓序列降采样率  
                
FTIP_TRK_X_START = 30                       # 图像有效区域切割
FTIP_TRK_X_END   = 290

FTIP_TRK_Y_START = 30
FTIP_TRK_Y_END   = 210

####################
# 其他参数
####################

import cv2
CV_CMAP_COLOR=cv2.COLORMAP_RAINBOW
# 以下是其他可选颜色
# COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_COOL, COLORMAP_HSV, COLORMAP_SPRING, COLORMAP_SUMMER
# COLORMAP_RAINBOW, COLORMAP_HOT, COLORMAP_JET, COLORMAP_OCEAN, COLORMAP_PINK, COLORMAP_WINTER
