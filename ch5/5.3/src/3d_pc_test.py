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
#from uv_texture import *
from axis import axis
import cal_normals
from icp import icp
# Constants
viewer=pyg_viewer_c()
opt=optimizer()
#####################PATH TO DATA###################
fname='./data/dep_1008-0.bin'
background='./data/dep_1008-2.bin'

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
    img_dep = IO.read_depth_image(fp)
    mask=filter.cube_filter(img_dep,img_dep_bg,distance=0.15)
    pc=dep_trans.depth_to_pcloud(img_dep,mask)
    show_pc(pc, viewer, T)
    pc_base=np.array([])
    pc_ref=np.copy(pc)
    pc_model=np.array([])
    cam_pos=np.zeros((int(max_count/3)+1,3))
    while True:
        if frame_count%T_count_interval==0 and frame_count<max_count :
            normals = cal_normals.compute_feature(pc,mean_k=8)#TODO smoth the normals
            normals,pc,pc_cut= filter.remove_edge_points(pc, normals,row_threshold=0.1,col_threshold=0.02)
            pc_with_n = np.hstack((pc, normals))
            if(frame_count>0):
                Tx, R, t= icp.cal_icp(pc_with_n[:,0:3],pc_model[:,1:7],T_global)#frame to model icp
            else:
                R=Tx[0:3,0:3]
                t=Tx[0:3,3]
            pose_count+=1
            cam_pos[pose_count]=axis.update_cam_pose(cam_pos[pose_count-1],Tx)
            print('R=',R,'\n','t=',t)
            global_theta,global_vec_n=opt.global_rot_axis(R,t)
            pc_base,T_global,pc_model=create_3D_pc(pc_base,pc_with_n,Tx,T_global,frame_count)
            pc_ref=np.copy(pc_with_n)

        elif frame_count==max_count:#reconstruct after optimization
            np.save('./tmp/cam_pos.npy',cam_pos)
            cam_center = cam_pos.mean(axis=0)
            facets,vertice=mesh.poission_rect(pc_base[:,1:4],pc_base[:,4:7],depth=7)
            IO.save_plyfile('./tmp/a.ply',vertice,np_facets=facets)
            IO.save_plyfile('./tmp/b.ply',pc_base[:,1:7])
            IO.save_plyfile('./tmp/d.ply', np.vstack((cam_center,cam_pos)))
        if frame_count < max_count:
            img_dep = IO.read_depth_image(fp)
            mask = filter.cube_filter(img_dep, img_dep_bg)
            pc = dep_trans.depth_to_pcloud(img_dep, mask)
        frame_count+=1

        show_pc(pc_base[:,1:4], viewer, T)

        #################user interface###################
        update_trans_mat=False
        evt,param=viewer.poll_evt()
        if evt is not None:
            print(evt)
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



if __name__ == "__main__":
    run()
