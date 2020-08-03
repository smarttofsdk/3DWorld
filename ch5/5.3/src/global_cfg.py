#!/usr/bin/python3
# coding=utf-8
import numpy as np
DEBUG_PRINT=False
TOF_TYPE='KINECT'#'NEW_TOF'
KINECT_SIM=True

CNT_SAV_MAX=600
KINECT_SIM_FNAME='kinect_data0'
KINECT_SAV_FNAME=KINECT_SIM_FNAME
KINECT_FPS=60

# 深度图尺寸（单位是像素）
KINECT_DEP_WID=512
KINECT_DEP_HGT=424
KINECT_DEP_SZ=KINECT_DEP_WID*KINECT_DEP_HGT

NEWTOF_DEP_WID=320
NEWTOF_DEP_HGT=240
NEWTOF_DEP_SZ=NEWTOF_DEP_WID*NEWTOF_DEP_HGT
# 红外图尺寸（单位是像素）
KINECT_IR_WID=512
KINECT_IR_HGT=424
KINECT_IR_SZ=KINECT_IR_WID*KINECT_IR_HGT

# 关节数据尺寸（单位是int16）
KINECT_JOINTS_SZ=50  # 25个关节，50个坐标值（x/y）

FRAME_IR_SZ=KINECT_IR_SZ
FRAME_DEP_SZ=KINECT_DEP_SZ
FRAME_JOINTS_SZ=KINECT_JOINTS_SZ

import cv2
CV_CMAP_COLOR=cv2.COLORMAP_RAINBOW
# 以下是其他可选颜色
# COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_COOL, COLORMAP_HSV, COLORMAP_SPRING, COLORMAP_SUMMER
# COLORMAP_RAINBOW, COLORMAP_HOT, COLORMAP_JET, COLORMAP_OCEAN, COLORMAP_PINK, COLORMAP_WINTER
if TOF_TYPE=='KINECT':
    IMG_WID=KINECT_DEP_WID
    IMG_HGT=KINECT_DEP_HGT
    IMG_SZ = IMG_WID * IMG_HGT
elif TOF_TYPE=='NEW_TOF':
    IMG_WID=NEWTOF_DEP_WID
    IMG_HGT=NEWTOF_DEP_HGT
    IMG_SZ=IMG_WID*IMG_HGT

TOF_CAM_ANGLE=0.887466864828949
TOF_CAM_EPS=1.0e-16

ax=ay=0.0                       # 点云旋转角度
cz=1.0                          # 点云旋转中心
mz=-0                           # 点云观察点位置
T = np.eye(4, dtype=np.float32)  # 点云变换矩阵
dmin=0.5    # 点云距离过滤门限（距离范围外的点被清除）
dmax=1.2
frame_cnt=0
mouse_down=False
mouse_x=mouse_y=0

N = 10                                    # number of random points in the dataset
num_tests = 1                                # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians)#  of the test set
K_rgb=np.matrix([[364.0706, 0, 311.1238],
        [0, 486.1341, 235.7895],
        [0, 0, 1]])
# 640*480
T_temp=np.matrix([-60.1376,
                  -0.8842,
                  19.1468])
# 640*480
K_ir=np.matrix([[360.8538, 0, 241.8416],
        [0, 361.1244, 203.6490],
        [0, 0, 1]])
# 640*480
R_ir2rgb=np.matrix([[1.0000, 0.0016, -0.0043],
        [-0.0015, 0.9999, 0.0113],
        [0.0043, -0.0114, 1.0006]])

#my variable
#max_count=144#icp iterate times
T_count_interval=3 #count T every 2 frames
register_interval=6 #mosaic new point cloud every 6 frames
import numpy as np
KINECT_DATA_TYPE=np.int16
NEWTOF_DATA_TYPE=np.float32