import numpy as np

class axis():
    def __init__(self,vector_n=[],position_n=[],theta=0):
        self.vector_n=vector_n
        self.position_n=position_n
        self.theta=theta
    def reset_axis_vector(self,vector_n):
        self.vector_n=vector_n

    def reset_axis_position(self, position_n):
        self.position_n = position_n

    def update_theta(self,theta):
        self.theta=theta


    def axis_calibration():#TODO 进行一系列处理（棋盘格）得到轴的位置和朝向向量 以及转角与相机位置的关系

        pass

    def cal_transform(self):
        R=self.vector_n*self.theta
        t=self.cal_translate(self.theta)
        T = np.identity(4)
        T[0:3, :] = np.hstack((R, t))
        return T,R,t

    def cal_translate(self,theta):#TODO 根据校准阶段获取的转角与相机位置的关系得到
        pass
        #return t
    @staticmethod
    def cal_transform(cls):#根据已经得到的轴的信息和转盘转角得到此帧相对世界坐标系的转移矩阵
        pass
    @staticmethod
    def update_cam_pose(last_pose,T):
        pose_tmp = np.ones(4)
        pose_tmp[0:3] = [0,0,0]
        current_pose = np.dot(T, pose_tmp.T).T
        return current_pose[0:3]
