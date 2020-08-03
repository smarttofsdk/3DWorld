import numpy as np
import time
import icp
import pygame
from depth_cam_tools import *
import sys,os
sys.path.append('./')
from global_cfg import *
from pyg_viewer import *
from pygame.locals import *

######################my function###########################

def create_3D_pc(pc_base,pc,T,T_global):
    T_global = np.dot(T_global, T)
    for item in pc:
        a=list(item)
        a.append(1)
        a=np.matrix(a)
        a=a.reshape((-1, 1))
        temp=T_global*a
        temp=temp.reshape((1, -1))
        temp=np.array(temp)
        temp=list(temp[0])
        pc_base=np.append(pc_base,temp[0:3])
        #pc_ref.append(temp[0:3])
    return pc_base.reshape((-1,3))

def T_to_Rt(icp_T)
    R=icp_T[0:3,0:3].T
    t=-icp_T[0:3,3]
    return R,t

def pc_to_base(pc,T):
    R,t=T_to_Rt(T)
    return np.dot(R,pc.T).T+t