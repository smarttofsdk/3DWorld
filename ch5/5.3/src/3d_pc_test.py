import numpy as np
import time
import pygame
import sys,os
sys.path.append('./')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2 as cv
from pc_trans_tool import *
from depth_cam_tools import *
from global_cfg import *
from pyg_viewer import *
from pygame.locals import *
from simple_3D_viewer import *
from pc_optimize_tools import *
from filter import filter
import IO
import mesh
from axis import axis
import cal_normals
from icp import icp
# Constants
viewer=pyg_viewer_c()
opt=optimizer()
#####################PATH TO DATA###################
fname='./data/depth_pig.bin'
background='./data/depth_pigback.bin'
###################################################
fp=open(fname,'rb')
fp_bg=open(background,'rb')
######################my function###########################
def run():
    ax=ay=0.0                       # 点云旋转角度
    cz=1.0                          # 点云旋转中心
    mz=-0                           # 点云观察点位置
    dmin=0.5    # 点云距离过滤门限（距离范围外的点被清除）
    dmax=1.2
    mouse_down=False
    mouse_x=mouse_y=0
    frame_count=0
    pose_count=0
    Tx=T=T_global=np.eye(4,dtype=np.float32)    # 点云变换矩阵
    img_dep_bg=IO.read_depth_image(fp_bg)
    while frame_count<20:
        img_dep = IO.read_depth_image(fp)
        frame_count+=1
    frame_count=0
    img_dep = IO.read_depth_image(fp)
    mask=filter.cube_filter(img_dep,img_dep_bg,distance=0.15)
    pc=dep_trans.depth_to_pcloud(img_dep,mask)
    show_pc(pc, viewer, T)
    pc_base=np.array([])
    pc_ref=np.copy(pc)
    pc_model=np.array([])
    max_count=330
    T_count_interval=5
    cam_pos=np.zeros((int(max_count/3)+1,3))
    while True:
        if frame_count%T_count_interval==0 and frame_count<max_count :
            print(frame_count)
            normals = cal_normals.compute_feature(pc,mean_k=8)#TODO smoth the normals
            normals,pc,pc_cut= filter.remove_edge_points(pc, normals,row_threshold=0.1,col_threshold=0.02)
            pc_with_n = np.hstack((pc, normals))
            if frame_count==0:
                pc_new=pc
                pc_cut=pc
            else:
                Tx, R, t= icp.cal_icp(pc_with_n[:,0:3],pc_model[:,1:7],T_global)#try frame to model icp
                pose_count+=1
                cam_pos[pose_count]=axis.update_cam_pose(cam_pos[pose_count-1],Tx)
                print('R=',R,'\n','t=',t)
                global_theta,global_vec_n=opt.global_rot_axis(R,t)
            pc_base,T_global,pc_model=create_3D_pc(pc_base,pc_with_n,Tx,T_global,frame_count)
            pc_ref=np.copy(pc_with_n)

        elif frame_count==max_count:#reconstruct after optimization
            np.save('./tmp/cam_pos.npy',cam_pos)
            cam_center = cam_pos.mean(axis=0)
            pc=pc_model[:,1:7]
            #remove noise points
            pc = filter.cubic_filter(pc,   -0.18,0.265,       0.02,0.35,     0.695,0.99)
            pc = filter.cubic_remove(pc,   -0.0,0.058,       0.24,0.276,     0.77,0.86)
            pc = filter.cubic_remove(pc,   -0.0069,0.052,       0.231,0.2406,     0.743,0.826)
            pc = filter.cubic_remove(pc,   0.009,0.04262,       0.228,0.231,     0.78,0.812)
            pc = filter.cubic_remove(pc,   -0.076,0.042,       0.29,0.37,     0.718,0.797)
            pc = filter.cubic_remove(pc,   0.007,0.094,       0.319,0.37,     0.874,0.892)
            pc = filter.cubic_remove(pc,   0.08,0.114,       0.304,0.334,     0.73,0.83)
            pc = filter.radius_outlier_removal(pc, 4, 0.005)
            facets,vertice=mesh.poission_rect(pc[:,0:3],pc[:,3:6],depth=6)
            IO.save_plyfile('./tmp/a.ply',vertice,np_facets=facets)
            IO.save_plyfile('./tmp/b.ply',pc[:,0:6])
            IO.save_plyfile('./tmp/d.ply', np.vstack((cam_center,cam_pos)))
            break
        if frame_count < max_count:
            img_dep = IO.read_depth_image(fp)
            mask = filter.cube_filter(img_dep, img_dep_bg)
            pc = dep_trans.depth_to_pcloud(img_dep, mask)
        frame_count+=1
        show_pc(pc_model[:,1:4], viewer, T)

if __name__ == "__main__":
    run()
