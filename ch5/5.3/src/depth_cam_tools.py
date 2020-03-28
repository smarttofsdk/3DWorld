#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import numpy as np
import struct
from global_cfg import *
# 装载私有库
sys.path.append('./')
from global_cfg      import *

## 弧度和角度互换
def deg_to_rad(d): return d*(np.pi/180.0)
def rad_to_deg(r): return r*(180.0/np.pi)


## 深度相机模拟器，负责点云变换及和深度图互换映射
#          Z轴
#   \     /|\     /
#    \     |     / 
#     \ Y  |    /
#  ----\--(.)--/----->  X轴
#       \  |  /
#       -\-+-/-  传感器平面
#         \|/
#         镜头，位置是：(cam_x,cam_y,cam_z)
# 注意：
#     点云以世界坐标定义的
#     相机坐标的轴和世界坐标方向一致
#     相机坐标的原点在世界坐标的位置是(cam_x,cam_y,cam_z)
class depth_cam_trans_c:
    def __init__(self):
        self.fp_sav=-1  # 用于保存深度帧的文件句柄
        self.cam_reset()
        return


    ## 相机位置、角度、镜头参数复位
    def cam_reset(self):
        self.cam_x=self.cam_y=self.cam_z=0.0                
        self.cam_ax=self.cam_ay=self.cam_az=0.0
        
        self.A=np.eye(4)
        self.B=np.eye(4)

        # 使用缺省的镜头参数，计算速算因子k和速算表
        self.cam_param_angle()  
        return


    ## 相机位置设置，即相机坐标的原点在世界坐标中的位置
    # 注意：相机坐标的轴和世界坐标方向一致
    def cam_mov(self,x=0.0,y=0.0,z=0.0):
        if DEBUG_PRINT: print('depth_cam_trans_c.cam_mov()')
        self.cam_x,self.cam_y,self.cam_z=x,y,z
        self.update_trans_mat()


    ## 改变相机角度，角度定义为镜头绕各个坐标轴的旋转角度
    # 注意：旋转绕着相机坐标，相机坐标的轴和世界坐标方向一致
    def cam_rot(self,ax=0.0,ay=0.0,az=0.0):
        if DEBUG_PRINT: print('depth_cam_trans_c.cam_rot()')
        self.cam_ax,self.cam_ay,self.cam_az=ax,ay,az
        self.update_trans_mat()


    ## 根据相机位置更新变换矩阵
    def update_trans_mat(self):
        if DEBUG_PRINT: print('depth_cam_trans_c.update_trans_mat()')
        # 先平移后选转
        A=np.eye(4)
        A[3,:]=(-self.cam_x,-self.cam_y,-self.cam_z,1.0)
        
        A=np.dot(A,pc_rotx(-self.cam_ax))  
        A=np.dot(A,pc_roty(-self.cam_ay))
        A=np.dot(A,pc_rotz(-self.cam_az))
        
        self.A=A
        self.B=np.array(np.mat(A).I)    # 反变换矩阵
        
        if DEBUG_PRINT: 
            print('A:'),; print(self.A)
            print('B:'),; print(self.B)
            print('A*B:'),;print(np.dot(self.A,self.B))


    ## 根据校准平面以及满幅视角数据重新计算k
    #
    #  |<--W_obj->| 校准物横向尺寸
    # _|__________|_______
    #  |\  视角   /| /|\
    #    \ angle/    | 
    #     \    /     |D_obj 校准物距离
    #     _\__/__    |
    #     __\/______\|/___
    def cam_param_angle(self, angle=deg_to_rad(TOF_CAM_ANGLE)):
        if DEBUG_PRINT: print('depth_cam_trans_c.cam_param_angle()')
        D_obj=1.0
        W_obj=2.0*D_obj*np.tan(angle/2.0);
        self.cam_param(W_obj,D_obj)
        return


    ## 根据满幅校准物体尺寸更新参数， 其中W_obj是视角校准物体水平长度，D_obj是校准物到镜头的距离
    # 注意：W_obj和D_obj可以分别是传感器物理尺寸和焦距距值
    def cam_param(self, W_obj=1.0, D_obj=1.0):
        if DEBUG_PRINT: print('depth_cam_trans_c.cam_param()')
        self.cam_param_k(W_obj/(IMG_WID*D_obj))
        return
    
    ## 根据参数k更新内部数据，注意：k=1/fx=1/fy
    def cam_param_k(self,k):
        self.k=k        # 像素位置(xp,yp)和物理坐标(x,y,z)的关系为：x=xp*k*z, y=yp*k*z
        self.gen_tab()  # 重新生成速算表

    ## 计算速算表
    #       tab_x[u,v]=(u-u0)*k
    #       tab_y[u,v]=(v-v0)*k
    # 其中：k=1/fx=1/fy, (u0,v0)是深度图中心的像素坐标
    # 通过速算表，计算像素位置(u,v)对应的物理坐标(x,y,z)
    #       x=tab_x[u,v]*z, y=tab_y[u,v]*z
    # 注意：为了方便使用，tab_x和tab_y矩阵被拉直成向量存放
    def gen_tab(self):
        if DEBUG_PRINT: print('depth_cam_trans_c.gen_tab()')
        u0=float(IMG_WID-1)*0.5
        v0=float(IMG_HGT-1)*0.5
        
        u=(np.arange(IMG_WID)-u0)*self.k
        v=(np.arange(IMG_HGT)-v0)*self.k
        
        self.tab_x=np.tile(u,IMG_HGT)
        self.tab_y=np.repeat(v,IMG_WID)
        return


    ## 从深度图img_dep计算点云，点云坐标(x,y,z)和像素位置(u,v)的对应关系为：
    #       x=img_dep[u,v]*(u-u_cent)/fx
    #       y=img_dep[u,v]*(v-v_cent)/fy
    #       z=img_dep[u,v]
    # 其中：(u-u_cent)/fx和(v-v_cent)/fy分别在速算表tab_x和tab_y中给出
    # 返回点云数据pc，每行是一个点的x/y/z坐标
    def depth_to_pcloud(self, img_dep, valid=[]):
        if len(valid)>0:
            sz=np.sum(valid.astype(int))
            pc=np.zeros((sz,3))
            pc[:,0]=img_dep[valid.flatten()].flatten()*self.tab_x[valid.flatten()]
            pc[:,1]=img_dep[valid.flatten()].flatten()*self.tab_y[valid.flatten()]
            pc[:,2]=img_dep[valid.flatten()].flatten()      
        else:
            pc=np.zeros((np.size(img_dep),3))
            pc[:,0]=img_dep.flatten()*self.tab_x
            pc[:,1]=img_dep.flatten()*self.tab_y
            pc[:,2]=img_dep.flatten()      
            
        return pc


    ## 将点云pc反向映射回到深度图。空间点(x,y,z)和像素位置(u,v)的对应关系为：
    #       u=x/(k*z)+u0
    #       v=y/(k*z)+v0
    # 返回深度图img_dep和像素有效信息valid
    # 如果入参output_idx==True的话，额外输出每个深度点对应的点云元素在pc中的行号
    def pcloud_to_depth(self,pc,output_idx=False):
        u0=float(IMG_WID-1)*0.5
        v0=float(IMG_HGT-1)*0.5

        # 计算点云投影到传感器的像素坐标
        x,y,z=pc[:,0],pc[:,1],pc[:,2]
        
        kz=self.k*z  # FIXME! kz的元素可能为0,意味着点云恰好落在镜头中心
        u=np.round(x/(kz+TOF_CAM_EPS)+u0).astype(int)
        v=np.round(y/(kz+TOF_CAM_EPS)+v0).astype(int)
        
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<IMG_WID)),
                             np.bitwise_and((v>=0),(v<IMG_HGT)))
        u_valid=u[valid]
        v_valid=v[valid]
        z_valid=z[valid]
        if output_idx: idx_valid=np.arange(len(u))[valid]
        
        # FIXME! need speedup
        # 将点云反向映射成深度图
        img_dep=np.full((IMG_HGT,IMG_WID),np.inf)        
        if output_idx:
            img_idx=np.full((IMG_HGT,IMG_WID),-1,dtype=int)        
            for ui,vi,zi,i in zip(u_valid,v_valid,z_valid,idx_valid):
                if zi<img_dep[vi,ui]:
                    img_dep[vi,ui]=zi   # 近距离像素屏蔽远距离像素
                    img_idx[vi,ui]=i    # 保存像素对应的点云序号
            valid=np.bitwise_and(~np.isinf(img_dep),img_dep>0.0)
            return (img_dep,img_idx,valid)
        else:
            for ui,vi,zi in zip(u_valid,v_valid,z_valid):
                img_dep[vi,ui]=min(img_dep[vi,ui],zi)   # 近距离像素屏蔽远距离像素
            valid=np.bitwise_and(~np.isinf(img_dep),img_dep>0.0)
            return (img_dep,valid)

    def pcloudone_to_depth(self,pc,rgb_img):
        T = K_rgb * T_temp.T
        R = K_rgb * R_ir2rgb * K_ir.I
        pc_idx=0
        dep_rgb=np.copy(pc)
        pc_coord=np.ones((len(pc),2))
        u0 = float(IMG_WID - 1) * 0.5
        v0 = float(IMG_HGT - 1) * 0.5
        for pt in pc:
           # 计算点云投影到传感器的像素坐标
            kz = self.k * pt[2]  # FIXME! kz的元素可能为0,意味着点云恰好落在镜头中心
            u = np.round(pt[0] / (kz + TOF_CAM_EPS) + u0).astype(int)
            v = np.round(pt[1] / (kz + TOF_CAM_EPS) + v0).astype(int)

            valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < IMG_WID)),
                                   np.bitwise_and((v >= 0), (v < IMG_HGT)))
            if valid == True:
                d = pt[2]
            else:
                d = 0
            uv_depth = np.matrix([u, v, 1])
            assert d != 0 and d != 65535
            uv_color = d * R * uv_depth.T + T / 1000#TODO limit the bound
            RGB_X=(uv_color[0]/uv_color[2]).astype(np.uint16)
            RGB_Y=(uv_color[1]/uv_color[2]).astype(np.uint16)
            dep_rgb[pc_idx]=rgb_img[RGB_Y,RGB_X]
            pc_coord[pc_idx]=np.array([(RGB_X)/640,(1-RGB_Y/480)]).reshape((1,2))
            #pc_coord[pc_idx] = np.array([() / 640, (1 - RGB_Y / 480)]).reshape((1, 2))
            pc_idx+=1

        return dep_rgb,pc_coord

    ## 将彩色点云pc反向映射回到彩色图。空间点(x,y,z)和像素位置(u,v)的对应关系为：
    #       u=x/(k*z)+u0
    #       v=y/(k*z)+v0
    # 输入：
    #       pc: 点云输组，每行对应一个点的x/y/z坐标
    #       color: 点云色彩，每行对应一个点的r/g/b值(uint8类型)
    # 返回彩色图img_rgb，深度图img_dep和像素有效信息valid
    def pcloud_to_depth_rgb(self,pc,color):
        u0=float(IMG_WID-1)*0.5
        v0=float(IMG_HGT-1)*0.5

        # 计算点云投影到传感器的像素坐标
        x,y,z=pc[:,0],pc[:,1],pc[:,2]
        
        kz=self.k*z  # FIXME! kz的元素可能为0,意味着点云恰好落在镜头中心
        
        u=np.round(x/kz+u0).astype(int)
        v=np.round(y/kz+v0).astype(int)
        
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<IMG_WID)),
                             np.bitwise_and((v>=0),(v<IMG_HGT)))
        u_valid=u[valid]
        v_valid=v[valid]
        z_valid=z[valid]
        idx_valid=np.arange(len(u))[valid]
        
        # FIXME! need speedup
        # 将点云反向映射成深度图和彩色图
        img_dep=np.full((IMG_HGT,IMG_WID),np.inf)
        img_rgb=np.zeros((IMG_HGT,IMG_WID,3),dtype=np.uint8)
        for ui,vi,zi,i in zip(u_valid,v_valid,z_valid,idx_valid):
            if zi<img_dep[vi,ui]:   # 近距离像素屏蔽远距离像素
                img_dep[vi,ui]=zi   
                img_rgb[vi,ui,:]=color[i,:]
        valid=np.bitwise_and(~np.isinf(img_dep),img_dep>0.0)
        return (img_dep,img_rgb,valid)

    ## 从相机深度图转换成点云，考虑了相机视角和坐标
    # 返回点云数据pc，每行是一个点的x/y/z坐标
    def cam_to_pcloud(self,img_dep,valid=[]): return pc_trans(self.B,self.depth_to_pcloud(img_dep,valid))

        
    ## 从点云转换成相机视角看到的深度图
    # 返回深度图img_dep和像素有效信息valid
    def pcloud_to_cam(self,pc): return self.pcloud_to_depth(pc_trans(self.A,pc))

    
    ## 打开文件，保存深度数据
    def open_file(self,fname='depth_frame.bin'):
        self.fp_sav=open(fname,'wb')

        
    # 保存深度数据帧
    def save(self,img_dep):
        if self.fp_sav<0: self.fp_sav=open('depth_frame.bin','wb')
        img_dep.flatten().astype(np.float32).tofile(self.fp_sav)

    
    ## 关闭深度数据文件
    def close_file(self):
        self.fp_sav.close()
        self.fp_sav=-1

## RGBD相机数据转换类
#     dep_cam rgb_cam     
#        [__]   [__]
#      __|__|___|__|___
#     |                |
#     +----------------+
#      以TOF相机光心为原点 
# 输入：
#   angle_dep: 深度相机的视角
#   angle_rgb: RGB相机的视角
#   rgb_x/y/z: RGB相机的光心坐标（TOF相机光心为原点）
#   rgb_ax,ay: RGB相机镜头绕x和y轴的旋转角度（单位是弧度）
class rgbd_cam_trans_c:
    def __init__(self,angle_dep,angle_rgb,rgb_x=0.0,rgb_y=0.0,rgb_z=0.0,rgb_ax=0.0,rgb_ay=0.0):
        self.dep_cam=depth_cam_trans_c()
        self.rgb_cam=depth_cam_trans_c()
    
        # 设置相机水平视角
        self.dep_cam.cam_param_angle(angle_dep)
        self.rgb_cam.cam_param_angle(angle_rgb)
        
        # 设置RGB相机的坐标和镜头朝向角（以TOF相机光心为原点）
        self.rgb_x=rgb_x
        self.rgb_y=rgb_y
        self.rgb_z=rgb_z
        self.rgb_ax=rgb_ax
        self.rgb_ay=rgb_ay
        return


    ## 从深度图和RGB图生成带颜色的彩色点云
    # 输入参数：mask是深度图的有效像素信息
    def calc_pcloud_rgb(self,img_dep,img_rgb,mask,filter_pc=False):
        pc=self.dep_cam.depth_to_pcloud(img_dep) # 深度图转换成点云
        
        # 计算相对于RGB相机的点云坐标(原点移到RGB相机中心)
        pc[:,0]-=self.rgb_x  
        pc[:,1]-=self.rgb_y
        pc[:,2]-=self.rgb_z
        
        # 计算相对于RGB相机的点云旋转(考虑RGB相机的镜头朝向角)
        T=np.dot(pc_rotx(-self.rgb_ax),pc_roty(-self.rgb_ay))
        pc=pc_trans(T,pc)
        
        # 计算RGB相机每个像素和点云对应关系
        dummy,img_idx,valid=self.rgb_cam.pcloud_to_depth(pc,output_idx=True)
        img_idx.shape=IMG_HGT,IMG_WID
        valid=valid.astype(bool).reshape(IMG_HGT,IMG_WID)       # valid是像素能够对应上的点
        
        mask=np.bitwise_and(valid,mask.reshape(IMG_HGT,IMG_WID))# 更新屏蔽码
        
        # 提取所有有效像素的像素坐标(u,v)以及对应点云的序号idx
        v,u=np.where(mask)
        idx=img_idx[v,u]
            
        # 提取颜色
        color=np.zeros((np.size(pc,0),3),dtype=np.uint8)
        color[idx,:]=img_rgb[v,u,:]
        
        # pc是从RGB相机角度看到的点云
        # color是点云对应的颜色
        if filter_pc:
            return pc[idx,:],color[idx,:]
        else:
            return pc,color

# RGBD相机对，基于像素匹配模型
class rgbd_cam_pair_c:
    def __init__(self,fname='./calib/calib.bin'):
        load_calib_data(fname)
        self.dep_cam=depth_cam_trans_c()
        return
        
    # 重新加载校准数据文件
    def load_calib_data(fname='./calib/calib.bin'):
        fid=open(fname,'r')
        
        # 第一组参数，RGB像素密集
        sz=np.fromfile(fid, dtype=int, count=1)[0]
        
        self.xr0=np.fromfile(fid, dtype=int, count=sz)
        self.yr0=np.fromfile(fid, dtype=int, count=sz)
        
        self.xt=np.fromfile(fid, dtype=int, count=sz)
        self.yt=np.fromfile(fid, dtype=int, count=sz)
        
        wx=np.fromfile(fid, dtype=np.float32,count=6)
        wy=np.fromfile(fid, dtype=np.float32,count=6)
    
        # 第二组参数，TOF像素密集
        sz=np.fromfile(fid, dtype=int, count=1)[0]
        
        self.xt0=np.fromfile(fid, dtype=int, count=sz)
        self.yt0=np.fromfile(fid, dtype=int, count=sz)
        
        self.xr=np.fromfile(fid, dtype=int, count=sz)
        self.yr=np.fromfile(fid, dtype=int, count=sz)
        
        ux=np.fromfile(fid, dtype=np.float32,count=6)
        uy=np.fromfile(fid, dtype=np.float32,count=6)    
        
        fid.close()
        return (xr0,yr0,xt,yt,wx,wy,xt0,yt0,xr,yr,ux,uy)
    
    # mask是TOF数据有效性图案，尺寸和深度图一样
    def calc_pcloud_rgb(self,img_dep,img_rgb,mask=[]):
        img_rgb_map=np.zeros((IMG_HGT,IMG_WID,3),dtype=np.uint8)
        img_rgb_map[self.yt0,self.xt0]=img_rgb[self.yr,self.xr]
        
        # 确定有效数据
        valid=np.full(False,(IMG_HGT,IMG_WID),dtype=bool)
        valid[self.yt0,self.xt0]=True   # 只有被填充过的数据是有效的
        if len(mask)>0: valid=np.bitwise_and(mask,valid)
        valid=valid.flatten()
        
        # 点运转换
        pc=self.dep_cam.depth_to_pcloud(img_dep)
        
        # 仅仅提取出有效数据
        pc=pc[valid,:]
        color=img_rgb_map[valid,:]
        return pc,color
        
        
## 将深度图归一化，转换成uint8矩阵
def depth_img_to_gray256(img,valid=[]):
    im_gray=img.copy()
    if len(valid)>0: im_gray[~valid]=0
    vmax=np.max(im_gray)
    if vmax>0: im_gray/=vmax
    return (im_gray*255).astype(np.uint8)


## 回放保存的深度数据
def play_back_depth_data(fname='depth_frame.bin'):
    fin=open(fname,'rb')
    
    import cv2       
    cv2.namedWindow("Image")   

    print('press "q" in GUI to quit')    
    while True:
        chunk=fin.read(4*IMG_SZ)
        if len(chunk)<4*IMG_SZ: break
        
        depth_img=np.array(struct.unpack('%df'%IMG_SZ, chunk))
        valid=np.bitwise_and(~np.isinf(depth_img),depth_img>0)
        
        im_gray=depth_img_to_gray256(depth_img,valid)
        
        cmap_img=cv2.applyColorMap(im_gray.reshape(IMG_HGT,IMG_WID), cv2.COLORMAP_RAINBOW)
        cv2.imshow("Image", cmap_img)
        
        key=cv2.waitKey(2)&0xFF
        if (key==ord('q')) or (key==ord(' ')): break
    
    fin.close()

    
## 伪色彩映射器（注意：可以用openCV的伪彩色变换替换，择机仅仅是后备显示方案）
class cmap_c:
    def __init__(self):
        self.mapR=np.hstack((np.ones(85),np.arange(84,-1,-1)/85.0,np.zeros(86),0.0))
        self.mapG=np.hstack((np.arange(85)/85.0,np.ones(86),np.arange(84,-1,-1)/85.0,0.0))
        self.mapB=np.hstack((np.zeros(85),np.arange(85)/85.0,np.ones(86),0.0))
        
    ## 计算图像的伪彩色映射
    def calc(self,img,valid=[]):
        img_int=np.clip(np.floor(img*255),0,255).flatten().astype(int)
        if np.size(valid)>0: img_int[~valid.flatten()]=256
        
        wid,hgt=img.shape
        imgRGB=np.zeros((wid,hgt,3))
        imgRGB[:,:,0]=self.mapR[img_int].reshape(img.shape)
        imgRGB[:,:,1]=self.mapG[img_int].reshape(img.shape)
        imgRGB[:,:,2]=self.mapB[img_int].reshape(img.shape)
        return imgRGB
    
    ## 计算归一化的图像的伪彩色映射
    def calc_norm(self,img,valid=[]):
        return self.calc(self.img_norm(img,valid),valid)
    
    ## 图像归一化
    def img_norm(self,img,valid=[]):
        vmax=np.max(img[valid]) if len(valid)>0 else np.max(img)
        vmin=np.min(img[valid]) if len(valid)>0 else np.min(img)
        if np.isinf(vmax) or np.isinf(vmin):
            return img
        else:    
            return (img-vmin)/(vmax-vmin) if vmax>vmin else (img-vmin) 


## 功能描述：
#     点云变换
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
def pc_trans(T,pc):      
    T_rot=T[0:3,0:3]        # 截取旋转部分
    pc_out=np.dot(pc,T_rot) # 计算旋转
    pc_out[:,0]+=T[3,0]     # 计算平移
    pc_out[:,1]+=T[3,1]
    pc_out[:,2]+=T[3,2]

    return pc_out


## 功能描述：
#     点云平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tx,ty,tz: 各个方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_mov(tx,ty,tz,pc=[]):    
    T=np.array([[ 1, 0, 0,0],
                [ 0, 1, 0,0],
                [ 0, 0, 1,0],
                [tx,ty,tz,1]])    # 移动坐标
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云沿x方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tx: 沿x方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_movx(tx,pc=[]): return pc_mov(tx,0,0,pc)


## 功能描述：
#     点云沿y方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     ty: 沿y方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_movy(ty,pc=[]): return pc_mov(0,ty,0,pc)    


## 功能描述：
#     点云沿z方向平移
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     tz: 沿z方向移动距离
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out点云变换类
def pc_movz(tz,pc=[]): return pc_mov(0,0,tz,pc)


## 功能描述：
#     点云沿着x轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotx(b,pc=[]):    
    T=np.array([[1,        0 ,       0 ,0],
                [0, np.cos(b),np.sin(b),0],
                [0,-np.sin(b),np.cos(b),0],
                [0,        0 ,       0 ,1]])    # 绕X轴旋转
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云沿着y轴旋转
# 用法：
#     [pc_out,T]=pc_roty(pc,b)
# 输入参数：
#     pc: 输入点云集depth_img合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵
def pc_roty(b,pc=[]):    
    T=np.array([[np.cos(b),0,-np.sin(b),0],
                [       0 ,1,        0 ,0],
                [np.sin(b),0, np.cos(b),0],
                [       0 ,0,        0 ,1]])    # 绕Y轴旋转
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数0,1)据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotz(b,pc=[]):    
    T=np.array([[ np.cos(b),np.sin(b),0,0],
                [-np.sin(b),np.cos(b),0,0],
                [        0 ,       0 ,1,0],
                [        0 ,       0 ,0,1]]);    # 绕Z轴旋转
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotx_mov(b,x,y,z,pc=[]):
    T1=pc_mov(-x,-y,-z)
    T2=pc_rotx(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)    
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_roty_mov(b,x,y,z,pc=[]):
    T1=pc_mov(-x,-y,-z)
    T2=pc_roty(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     点云以(x,y,z)为中心，沿着z轴旋转
# 输入参数：test_cam_cv()
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     b:  转动角度点云变换类
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 注意：
#     如果pc=[]，则返回T，否则返回pc_out
def pc_rotz_mov(b,x,y,z,pc=[]):
    T1=pc_mov(-x,-y,-z)
    T2=pc_rotz(b)
    T3=pc_mov(x,y,z)
    T=np.dot(np.dot(T1,T2),T3)
    return pc_trans(T,pc) if len(pc) else T


## 功能描述：
#     生成圆柱体点云
# 输入参数：
#     L: 长度
#     R: 半径（第一轴）
#     KX:X方向抽样数量
#     KA:角度抽样数量
#     KS:扁平度
# 输出参数：
#     pc：输出点云集合，每个点对应一行数据坐标(x,y,z)
def make_cylinder(L=1.0,R=0.5,KX=80.0,KA=200.0,KS=2.0):

    # 生成圆柱体点云
    vec_x=np.arange(KX)/KX-0.5
    vec_a=np.arange(KA)/KA*2.0*np.pi;
    
    pc_x,pc_a=np.meshgrid(vec_x,vec_a)
    
    pc_x.shape=np.size(pc_x),1
    pc_a.shape=np.size(pc_a),1
    
    pc_y=np.sin(pc_a)*R*KS
    pc_z=np.cos(pc_a)*R
    
    pc=np.hstack((pc_x*L,pc_y,pc_z))
    return pc


## 功能描述：
#     生成矩形平面点云, 矩形位置和XY平面重合，中心在原点
# 输入参数：
#     H,W: 矩形高宽
#     KH,KW: 矩形长宽方向的采样点数量
# 输出参数：
#     pc：输出点云集合，每个点对应一行数据坐标(x,y,z)
def make_rectangle(H=1.0,W=2.0,KH=50.0,KW=100.0):

    vec_y=(np.arange(KH)/float(KH)-0.5)*H
    vec_x=(np.arange(KW)/float(KW)-0.5)*W
    
    pc_x,pc_y=np.meshgrid(vec_x,vec_y)
    
    pc_x.shape=np.size(pc_x),1
    pc_y.shape=np.size(pc_y),1
    pc_z=np.zeros_like(pc_x)

    return np.hstack((pc_x,pc_y,pc_z))


## 功能描述：
#     生成立方体, 立方体中心在原点
# 输入参数：
#     H,W,D: 立方体高宽深
#     KH,KW,KD: 立方体长宽深方向的采样点数量
# 输出参数：
#     pc：输出点云集合，每个点对应一行数据坐标(x,y,z)
def make_cubic(H=1.0,W=1.5,D=2.0,KH=25,KW=50,KD=75):
    pc_left =make_rectangle(H,D,KH,KD)
    pc_up   =make_rectangle(D,W,KD,KW)
    pc_front=make_rectangle(H,W,KH,KW)
    
    pc_left=pc_roty(deg_to_rad(90.0),pc_left)
    pc_up  =pc_rotx(deg_to_rad(90.0),pc_up)
    
    pc_right=pc_left
    pc_down =pc_up
    pc_back =pc_front
    
    pc_left =pc_mov(-W/2.0,0,0,pc_left )
    pc_right=pc_mov( W/2.0,0,0,pc_right)
    
    pc_up   =pc_mov(0,-H/2.0,0,pc_up   )
    pc_down =pc_mov(0, H/2.0,0,pc_down )
    
    pc_front=pc_mov(0,0,-D/2.0,pc_front)
    pc_back =pc_mov(0,0, D/2.0,pc_back )
    
    return np.vstack((pc_left,pc_right,pc_up,pc_down,pc_front,pc_back))


## 功能描述：
#     生成球体点云（用随机采样）
# 输入参数：
#     R: 第一轴半径
#     K: 随机采样数量
#     KS1,KS2:扁平度
# 输出参数：
#     pc：输出点云集合，每个点对应一行数据坐标(x,y,z)
def make_sphere(R=1.0,K=8000,KS1=1.5,KS2=0.5):
    a1=np.random.rand(int(K),1)*np.pi*2.0
    a2=np.random.rand(int(K),1)*np.pi*2.0
    
    s1=np.sin(a1)
    c1=np.cos(a1)
    
    s2=np.sin(a2)
    c2=np.cos(a2)
    
    pc1=np.hstack((s1*s2, c1,s1*c2))
    pc2=np.hstack((c1*s2,-s1,c1*c2))
    pc=np.vstack((pc1,pc2))
    
    pc[:,0]*=R
    pc[:,1]*=R*KS1
    pc[:,2]*=R*KS2
    
    return pc
    

## 功能描述：
#       通过matplotlib显示3D点云
# 输入参数：
#   fig: 图片窗口，如果不提供的化，缺省选择figure(1)
#   xlabel,ylabel,zlabel: 坐标轴显示内容，如果不提供，缺省显示'X/Y/Z'
#   xlim,ylim,zlim: 人为调整坐标显示范围，如果不提供，则不进行坐标轴显示范围调整
def pc_plot3d(pc,fig=None,xlabel='X',ylabel='Y',zlabel='Z',xlim=(),ylim=(),zlim=()):
    if fig is None: 
        fig = plt.figure(1)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0],pc[:,1],pc[:,2])
    
    if len(xlabel): ax.set_xlabel(xlabel)  
    if len(ylabel): ax.set_ylabel(ylabel)  
    if len(zlabel): ax.set_zlabel(zlabel)  
    
    if len(xlim): ax.set_xlim(xlim)
    if len(ylim): ax.set_ylim(ylim)
    if len(zlim): ax.set_zlim(zlim)
    
    return

## 罗德里格旋转公式求旋转矩阵
def calc_rot_mat(vectsrc,vectdst):
    angle=np.acos(np.sum(vectsrc*vectdst))
    vector=np.cross(vectsrc,vectdst);
    vector/=np.sqrt(np.sum(vector**2)) # normalization
    cost=np.cos(angle)
    sint=np.sin(angle)
    x,y,z=vector[0],vector[1],vector[2]

    # rotate matrix
    T=np.array([[x*x*(1-cost)+cost,  (1-cost)*x*y-sint*z,(1-cost)*x*z+sint*y],
                [(1-cost)*y*x+sint*z,(1-cost)*y*y+cost  ,(1-cost)*y*z-sint*x],
                [(1-cost)*z*x-sint*y,(1-cost)*z*y+sint*x,(1-cost)*z*z+cost  ]])

    return T


########################################
# 以下是辅助函数
########################################

## 通过模拟深度相机模拟器在屏幕上显示点云
def cam_show(pc_view, cam_x=0.0,cam_y=0.0,cam_z=-1.0,rot_x=0.0,rot_y=0.0,rot_z=0.0,cam_sim=None):
    if not cam_sim:
        cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
        cam_sim.cam_mov(cam_x,cam_y,cam_z)    
    
    import cv2   
    cv2.namedWindow("Image")   
    
    print('press "q" in GUI to quit')
    while True:
        if (rot_x!=0) or (rot_y!=0) or (rot_z!=0):
            T=np.dot(np.dot(pc_rotx(rot_x),pc_roty(rot_y)),pc_rotz(rot_z))
            pc_view=pc_trans(T,pc_view)
        
        (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
        im_gray=depth_img_to_gray256(depth_img,valid)
        cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
        cv2.imshow("Image", cmap_img)
        key=cv2.waitKey(1)&0xFF
        if (key==ord('q')) or (key==ord(' ')): break
    return

########################################
# 以下是测试代码
########################################

## 测试点云生成、运动和显示(使用matplotlib)
def test_pc():
    pc=make_cylinder()          # 建立点云数据pc
    print('size of point cloud:%d'%len(pc))
    
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
    cmap=cmap_c()

    # 点云变换
    pc_view=pc_mov(0,0,2,pc)
    pc_view=pc_rotz_mov(deg_to_rad(30),0,0,2,pc_view)
    pc_view=pc_roty_mov(deg_to_rad(30),0,0,2,pc_view)

    pc_plot3d(pc=pc_view,          
              fig=plt.figure(1),
              xlim=(-2.0,2.0),
              ylim=(-2.0,2.0),
              zlim=(-2.0,2.0))
    
    # 从深度图反向映射回到点云（受相机视角和遮挡，应该有可能缺失点云）
    (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
    cmap_img=cmap.calc_norm(depth_img,valid)
    
    plt.figure(2)
    plt.imshow(cmap_img)
    plt.show()


## 测试相机运动和点云显示(使用matplotlib)
def test_cam():
        
    pc_view=make_cylinder(L=1.0,R=0.5,KX=200.0,KA=500.0,KS=2.0)     # 建立点云数据pc
    pc_plot3d(pc=pc_view,          
              fig=plt.figure(1),
              xlim=(-2.0,2.0),
              ylim=(-2.0,2.0),
              zlim=(-2.0,2.0))
              
    cmap=cmap_c()
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
    cam_sim.cam_mov(0,0,-2.5)
    cam_sim.cam_rot(deg_to_rad(10.0),deg_to_rad(10.0),deg_to_rad(10.0))

    
    (img_dep,valid)=cam_sim.pcloud_to_cam(pc_view)
    img_cmap=cmap.calc_norm(img_dep,valid)
    
    plt.figure(2)
    plt.imshow(img_cmap)
    
    pc_cam=cam_sim.cam_to_pcloud(img_dep,valid)
    pc_plot3d(pc=pc_cam,          
              fig=plt.figure(3),
              xlim=(-2.0,2.0),
              ylim=(-2.0,2.0),
              zlim=(-2.0,2.0))
    
    plt.show()
    

## 测试点云生成、运动和显示(使用opencv)
def test_pc_cv():
        
    pc=make_cylinder()          # 建立点云数据pc
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
    pc_view=pc_mov(0,0,2,pc)    # 点云平移
    
    import cv2   
    from time import clock
    
    cv2.namedWindow("Image")   
    start=clock()
    n=0
    print('press "q" in GUI to quit')
    while True:
        n+=1
        pc_view=pc_rotz_mov(deg_to_rad(1),0,0,2,pc_view)
        pc_view=pc_roty_mov(deg_to_rad(3),0,0,2,pc_view)
    
        (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
        
        im_gray=depth_img_to_gray256(depth_img,valid)
        
        #cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
        cv2.imshow("Image", cmap_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    finish=clock()
    print('run time: %f(ms)'%(1000.0*(finish-start)/float(n)))


## 测试相机运动和点云显示(使用opencv)
def test_cam_cv():        
    pc_view=make_cylinder()     # 建立点云数据pc
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器

    import cv2   
    from time import clock
    
    cv2.namedWindow("Image")   
    start=clock()
    n=0
    
    rot_y=0.0
    R=3.0
    print('press "q" in GUI to quit')
    while True:
        n+=1
        rot_y+=deg_to_rad(1)
        rot_y%=np.pi*2

        cam_sim.cam_rot(0,rot_y,0)    
        cam_sim.cam_mov(-np.sin(rot_y)*R,0,-np.cos(rot_y)*R)
    
        (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
        
        im_gray=depth_img_to_gray256(depth_img,valid)
        
        cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
        cv2.imshow("Image", cmap_img)
        
        key=cv2.waitKey(1)&0xFF
        if (key==ord('q')) or (key==ord(' ')): break
    
    finish=clock()
    print('run time: %f(ms)'%(1000.0*(finish-start)/float(n)))


## 测试不同物件点云生成、运动和显示(使用opencv)
def test_obj_cv():
        
    # 生成不同物体点云
    pc0=pc_mov(0,0,5,make_cylinder())
    pc1=pc_mov(0,0,5,make_rectangle())
    pc2=pc_mov(0,0,5,make_cubic())
    pc3=pc_mov(0,0,5,make_sphere())
    
    pc_list=[pc0,pc1,pc2,pc3]
    
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
    
    import cv2   
    from time import clock
    
    cv2.namedWindow("Image")   
    start=clock()
    pc_view=pc0
    n=0
    print('press "q" in GUI to quit')
    while True:
        n+=1
        if n%100==0:
            idx=int(n/100)%len(pc_list)
            pc_view=pc_list[idx]
        
        pc_view=pc_rotz_mov(deg_to_rad(1),0,0,5,pc_view)
        pc_view=pc_roty_mov(deg_to_rad(3),0,0,5,pc_view)
        pc_view=pc_rotx_mov(deg_to_rad(5),0,0,5,pc_view)
    
        (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
        
        im_gray=depth_img_to_gray256(depth_img,valid)
        
        #cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cmap_img=cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
        cv2.imshow("Image", cmap_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    finish=clock()
    print('run time: %f(ms)'%(1000.0*(finish-start)/float(n)))


## 测试相机保存深度数据帧和回放(使用opencv)
def test_cam_save_playback():        
    pc=make_cylinder()          # 建立点云数据pc
    cam_sim=depth_cam_trans_c()   # 建立深度相机模拟器
    pc_view=pc_mov(0,0,3,pc)    # 点云平移

    print('generating depth frames ...')
    # 打开文件准备保存
    cam_sim.open_file()
    for n in range(500):
        pc_view=pc_rotz_mov(deg_to_rad(1),0,0,3,pc_view)
        pc_view=pc_roty_mov(deg_to_rad(3),0,0,3,pc_view)
    
        (depth_img,valid)=cam_sim.pcloud_to_cam(pc_view)
        
        # 保存点云帧
        cam_sim.save(depth_img)
    cam_sim.close_file()
    
    # 回放保存的图像帧
    print('playback depth frames')
    play_back_depth_data()


## 测试RGBD相机变换类
def test_rgbd_cam_trans(use_cv=True):
    
    # 参数设置
    angle_dep=deg_to_rad(60)
    angle_rgb=deg_to_rad(60)    # 90Deg
    rgb_x=-0.55                 # +1.55
    rgb_y=0.0
    rgb_z=0.0
    rgb_ax=deg_to_rad(0.0)
    rgb_ay=deg_to_rad(0.0)
    
    # 创建相机对象，设定参数
    rgbd_cam=rgbd_cam_trans_c(angle_dep,angle_rgb,rgb_x,rgb_y,rgb_z,rgb_ax,rgb_ay)
                              
    # 创建点云对象
    pc0=pc_mov(0,0,2.732,pc_rotz(deg_to_rad( 20),make_rectangle(H=1.0,W=1.5,KH=200.0,KW=300.0)))
    pc1=pc_mov(1,0,3.732,pc_roty(deg_to_rad(-30),make_rectangle(H=1.0,W=1.5,KH=100.0,KW=150.0)))
    
    num0=np.size(pc0,0)
    num1=np.size(pc1,0)
    
    # 创建点云颜色矩阵
    color0=np.tile(np.array([  0,255,255],dtype=np.uint8),num0).reshape(num0,3)
    color1=np.tile(np.array([255,  0,255],dtype=np.uint8),num1).reshape(num1,3)
    
    # 点云合并
    pc=np.vstack((pc0,pc1))
    color=np.vstack((color0,color1))

    # 生成深度图
    (img_dep,valid)=rgbd_cam.dep_cam.pcloud_to_cam(pc)
    
    # 生成深度相机看到的伪彩色图
    cmap=cmap_c()
    img_cmap=cmap.calc_norm(img_dep,valid)
    
    if not use_cv:
        plt.figure(1)
        plt.imshow(img_cmap)
    
    # 生成RGB相机看到的彩色图
    pc_rgb=pc.copy()
    
    # 计算相对于RGB相机的点云坐标(原点移到RGB相机中心)
    pc_rgb[:,0]-=rgb_x  
    pc_rgb[:,1]-=rgb_y
    pc_rgb[:,2]-=rgb_z
    
    # 计算相对于RGB相机的点云旋转(考虑RGB相机的镜头朝向角)
    T=np.dot(pc_rotx(-rgb_ax),pc_roty(-rgb_ay))
    pc_rgb=pc_trans(T,pc_rgb)
    
    dummy,img_rgb,valid_rgb=rgbd_cam.rgb_cam.pcloud_to_depth_rgb(pc_rgb,color)
    
    img_rgb[~valid_rgb,:]=0
    if not use_cv:
        plt.figure(2)
        plt.imshow(img_rgb)
    
    # 对点云染色
    pc_rgbd,color_rgbd=rgbd_cam.calc_pcloud_rgb(img_dep,img_rgb,valid)
    
    if not use_cv:
        pc_plot3d(pc=pc_rgbd,          
                fig=plt.figure(3),
                xlim=(-2.0,2.0),
                ylim=(-2.0,2.0),
                zlim=( 0.0,5.0))
              
    if not use_cv: 
        plt.show()
        return

    # 旋转显示染色的点云
    import cv2   
    from time import clock
    
    cv2.namedWindow("Image")   
    pc_view=pc_rgbd
    
    print('press "q" in GUI to quit')
    while True:        
        pc_view=pc_rotz_mov(deg_to_rad(1),0,0,3,pc_view)
        pc_view=pc_roty_mov(deg_to_rad(3),0,0,3,pc_view)
        pc_view=pc_rotx_mov(deg_to_rad(1),0,0,3,pc_view)
    
        # 生成RGB相机看到的彩色图
        dummy,img_rgb,valid_rgb=rgbd_cam.rgb_cam.pcloud_to_depth_rgb(pc_view,color_rgbd)
        cv2.imshow("Image", img_rgb)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    return
    
####################
# 测试不同功能
####################

# 测试点云生成，通过matplotlib显示
#test_pc()

# 测试点云生成，通过openCV显示
#test_pc_cv()

# 测试相机移动观察点云，通过matplotlib显示
#test_cam()

# 测试相机移动观察点云，通过openCV显示
#test_cam_cv()

# 测试不同形状点云生成，通过openCV显示
#test_obj_cv()

# 测试点云生成以及深度图保存和回放，通过openCV显示
#test_cam_save_playback()

# 测试RGBD融合算法
#test_rgbd_cam_trans()
