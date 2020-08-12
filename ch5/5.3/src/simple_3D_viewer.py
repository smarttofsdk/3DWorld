#!/usr/bin/python3
# coding=utf-8

## 简易3D数据浏览器

import ctypes
import cv2
import pygame
import sys,os
import numpy as np
import time
from icp import icp
# 装载私有库
sys.path.append('./')
from global_cfg import *
from pyg_viewer import *
from depth_cam_tools import *
from pygame.locals import *
import filter
delay=1.0/KINECT_FPS

#viewer=pyg_viewer_c()

# 深度图处理工具
dep_trans=depth_cam_trans_c()
dep_trans.cam_param_k(1.0/3.72e2)

# 点云变换参数

######################my function###########################
# def get_bg_pc(fp_bg):#get background point cloud
#     frame_bg=np.fromfile(fp_bg,dtype=np.float32,count=FRAME_DEP_SZ)
#     dep_bg=frame_bg.copy()#.stype(np.float32)/1000.0
#     bg_pc=dep_trans.depth_to_pcloud(dep_bg)
#     return bg_pc,dep_bg

# def get_rect(dep_image,start_x,start_y,width,height):#  get a rectangle of source image
#     rect_dep_image=np.zeros(FRAME_DEP_SZ)
#     for i in np.arange(height):
#         rect_dep_image[(start_y+i)*KINECT_DEP_WID+start_x:(start_y+i)*KINECT_DEP_WID+start_x+width]=dep_image[(start_y+i)*KINECT_DEP_WID+start_x:(start_y+i)*KINECT_DEP_WID+start_x+width]
#     return rect_dep_image


def show_pc(pc,viewer,T=np.eye(4,dtype=np.float32)):
    # 点云变换，并将变换后的点云映射回深度图
    pc_new=pc_trans(T, pc)
    img_dep_new,mask=dep_trans.pcloud_to_depth(pc_new)

    # 将深度图转换成伪彩色
    img_norm=np.clip(np.float32(img_dep_new - dmin)/np.float32(dmax-dmin),0.0,1.0)
    img_u8=np.uint8(img_norm*255)
    img_rgb=cv2.applyColorMap(255-img_u8,CV_CMAP_COLOR)
    img_rgb[~mask,:] =0
    # 点云显示
    viewer.update_pan_img_rgb(img_rgb)
    viewer.update()
# def img_with_filter_pc_from_file(fp,img_dep_bg):
#     frame = np.fromfile(fp,dtype=np.int16,count=FRAME_DEP_SZ)
#     if len(frame)<FRAME_DEP_SZ:
#         fp.seek(0, os.SEEK_SET)
#         frame=np.fromfile(self.sim_file_dep,dtype=np.int16,count=FRAME_DEP_SZ)
#
#     img_dep = frame.copy().astype(np.float32) / 1000.0  # 注意，单位是M
#     img_dep = cv2.bilateralFilter(img_dep, 5, 3, 34)
#     img_dep = get_rect(list(img_dep), 100, 100, 200, 250)  # wood
#     img_dep = img_dep.copy().astype(np.float32).reshape([424, 512])
#     #blur = my_biFil(img_dep, 3)
#     img_blur = blur.copy().astype(np.float32).reshape([1,-1])
#     # img_dep = get_rect(img_dep, 150, 150, 150, 120)#cube
#     # img_dep = get_rect(img_dep, 180, 150, 80, 100)#clock
#     mask = get_mask(img_blur, img_dep_bg)  # mask选出距离范围[dmin,dmax]的深度图像素
#     img_blur = blur.copy().astype(np.float32).reshape([-1, 1])
#     # mask = mask.flatten()
#     pc = dep_trans.depth_to_pcloud(img_blur, mask)
#     return pc
# def pc_from_file(fp,img_dep_bg,CUT_IMG=True):
#    frame=np.fromfile(fp,dtype=np.float32,count=FRAME_DEP_SZ)
#     if len(frame)<FRAME_DEP_SZ:
#         fp.seek(0, os.SEEK_SET)
#         frame=np.fromfile(self.sim_file_dep,dtype=np.int16,count=FRAME_DEP_SZ)
#         #frame_cnt+=1
#     # 将深度图转换成点云
#     img_dep=frame.copy()#.astype(np.float32)/1000.0  # 注意，单位是M
#     #img_dep = list(cv2.bilateralFilter(img_dep, 5, 3, 34))
#     if CUT_IMG==True:
#         #img_dep=get_rect(img_dep,100,100,200,250)#wood
#         img_dep = get_rect(img_dep,50, 50, 150, 150)#new tof
#         # img_dep = get_rect(img_dep, 150, 150, 150, 120)#cube
#         # img_dep = get_rect(img_dep, 180, 150, 80, 100)#clock
#     mask=get_mask(img_dep,img_dep_bg)# mask选出距离范围[dmin,dmax]的深度图像素
#     pc=dep_trans.depth_to_pcloud(img_dep,mask)
#
#     #pc=filter.remove_isolation(pc,std_dev=2)
#     return pc,mask

# def get_mask(img_dep,img_dep_bg,ccc=0.5,dmax=1.2):#reserve pc between 0.5m~1.2m
#     return (img_dep>dmin)*(img_dep<dmax)*(np.abs(img_dep_bg-img_dep)>0.2)

