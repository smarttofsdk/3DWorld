import numpy as np
import math
import cv2
import pc_trans_tool
import filter
from simple_3D_viewer import *
T=np.eye(4,dtype=np.float32)
viewer=pyg_viewer_c()
#pc=pc_from_file(fp,img_dep_bg)
class reconstructor():
    def __init__(self,T_optimized,T_count_interval,mosaic_interval):
        self.T_count_interval=T_count_interval
        self.mosaic_interval=mosaic_interval#多少帧拼接一次
        self.T_global=np.array([])
        self.T_optimized=T_optimized
        self.frame_cnt=0
        self.pc_base=np.array([])
        self.key_frame=np.array([])
    def reconstruct(self,fp,img_dep_bg):
        self.T_global=self.get_T_global()
        pc = pc_from_file(fp, img_dep_bg)
        for i in range(self.mosaic_interval*self.T_global.shape[0]):
            pc=pc_from_file(fp,img_dep_bg)#cut first frame
            if(i%(self.mosaic_interval)==0):
                j=int(i/(self.mosaic_interval))
                #T_global = np.dot(T_global,T)
                pc_tmp=np.ones((pc.shape[0],4))
                pc_tmp[:,0:3] = np.copy(pc)
                pc_tmp=np.dot(self.T_global[j],pc_tmp.T).T
                #update the final point cloud every six times
                self.pc_base=np.append(self.pc_base,pc_tmp[:,0:3])
                #show_pc(self.pc_base.reshape((-1,3)),viewer)
        self.pc_base=self.pc_base.reshape((-1,3))
        #show_pc(self.pc_base,viewer)
        self.pc_base=filter.remove_outlier(self.pc_base,std_dev=0.1)
        return self.pc_base

    def get_T_global(self):#Global T of keyframe
        #j=0
        T_global_tmp=np.identity(4)
        for i in range(self.T_optimized.shape[0]):
            T_global_tmp=np.dot(T_global_tmp,self.T_optimized[i])
            if (i*self.T_count_interval)%self.mosaic_interval==0:
                self.T_global=np.append(self.T_global,T_global_tmp)
        return self.T_global.reshape((-1,4,4))

    def get_key_frame(self,fp,img_dep_bg):#TODO frame used for mosaic the global point cloud
        #key_frame_count=0
        pass
        #return self.key_frame