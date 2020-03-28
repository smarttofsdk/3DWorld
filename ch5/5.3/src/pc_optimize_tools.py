import numpy as np
import math
import cv2
import pc_trans_tool
from icp import icp
class optimizer():
    def __init__(self):
        self.global_axis_vec=np.zeros((3,1))
        self.global_axis_theta=0
        self.global_ave_vec = np.zeros((3, 1))
        self.global_ave_theta = 0
        self.count=0
        self.T_optimized=np.identity(4)
    def single_rot_axis(self,R,t):#每次旋转轴
        #theta=math.acos((np.trace(R)-1)/2)
        vec_n=np.zeros((3,1))
        cv2.Rodrigues(R,vec_n)
        theta=np.linalg.norm(vec_n)
        vec_n=vec_n/theta
        if(vec_n[1]<0):
            vec_n=-vec_n
        #theta,vec_n=self.global_rot_axis(vec_n,theta)
        #print('theta=',theta,'\n vector n =',vec_n,'\n trans_n=',t)
        return theta,vec_n,t

    def global_rot_axis(self,R,t):#全局平均旋转轴
        theta,vec_n,_=self.single_rot_axis(R,t)
        self.count+=1
        self.global_axis_vec+=vec_n
        self.global_ave_vec=self.global_axis_vec/self.count
        self.global_axis_theta+=theta
        self.global_ave_theta=self.global_axis_theta/self.count
        #print('global theta=',self.global_ave_theta,'\n global vector n =',self.global_ave_vec)
        print('\n global vector n =', self.global_ave_vec)
        # R = self.transback_to_R(self.global_ave_vec)
        # global_ave_T= pc_trans_tool.Rt_to_T(R, t)
        return self.global_ave_theta,self.global_ave_vec

    def modify_R_t(self,R,t):#
        theta,vect_n,_=self.single_rot_axis(R,t)
        modified_R=self.transback_to_R(vect_n)
        T=pc_trans_tool.Rt_to_T(modified_R,t)
        self.T_optimized=np.append(self.T_optimized,T).reshape((-1,4,4))
        #print('modified T=\n',T,'\n')
        return T,self.T_optimized

    def transback_to_R(self,vect_n):#轴角变回旋转矩阵
        R=np.identity(3)
        cv2.Rodrigues(vect_n,R)#*(theta/np.linalg.norm(vect_n))
        #print('R modified to:\n', R, '\n')
        return R

    def check_loopclosure(self,src,key_frames,threshold):
        loopclosure_flag=False
        for i in range(len(key_frames)):
            error=self.cal_likelihood(src,key_frames[i,:])
            if(error<threshold):
                loopclosure_flag=True
                return loopclosure_flag

    def cal_likelihood(self,src,key_frame):
        distances, _ = icp.nearest_neighbor(src,key_frame)
        mean_error = np.mean(distances)
        return mean_error