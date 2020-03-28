import numpy as np
from filter import filter
######################my function###########################

def create_3D_pc(pc_base,pc_with_n,T,T_global,count,opt=None):#TODO:optimizer will be applied later
    # T_global = np.dot(T_global,T)
    T_global=T
    pc_model=pc_to_model(pc_base,pc_with_n,T_global,count)
    if count%6==0:#update the final point cloud every 9 times
        pc_base=pc_model
    return pc_base,T_global,pc_model#,pc_tmp[:,0:3]

def pc_to_model(pc_model,pc_with_n,T_global,count):
    pc = pc_with_n[:, 0:3]
    normals = pc_with_n[:, 3:6]
    # normals=cal_normals.compute_feature(pc, mean_k=5)
    # normals,pc,pc_cut=filter.remove_edge_points(pc,normals,threshold=0.3)
    pc_tmp = np.ones((pc.shape[0], 4))
    pc_tmp[:, 0:3] = np.copy(pc)
    normal_tmp = np.ones((normals.shape[0], 4))
    normal_tmp[:, 0:3] = np.copy(normals)
    pc_tmp = np.dot(T_global, pc_tmp.T).T
    normal_tmp = np.dot(T_global, normal_tmp.T).T
    pc_with_n = np.hstack(((count*np.ones((len(pc_tmp),1))).astype('int'),pc_tmp[:, 0:3], normal_tmp[:, 0:3]))
    pc_model = np.append(pc_model, pc_with_n)
    pc_model = pc_model.reshape((-1, 7))
    pc_model=filter.remove_isolation(pc_model, std_dev=1)
    return pc_model

def T_to_Rt(icp_T):
    R=icp_T[0:3,0:3].T
    t=icp_T[0:3,3]
    return R,t

def Rt_to_T(R,t):
    T=np.identity(4)
    T[0:3,:]=np.hstack((R,t))
    #T[0:3,3]=t
    return T

def T_transform(pc,T):
    pc_tmp = np.ones((pc.shape[0], 4))
    pc_tmp[:, 0:3] = np.copy(pc[:, 0:3])
    pc_tmp = np.dot(T, pc_tmp.T).T
    return pc_tmp[:, 0:3]

def pc_to_base(pc,T):
    R,t=T_to_Rt(T)
    return np.dot(R,pc.T).T+t

