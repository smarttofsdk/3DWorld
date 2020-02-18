#!/usr/bin/python3
# coding=utf-8

import cv2
import pygame
import sys, os, time

import numpy as np
import pylab as plt
# import zmq

sys.path.append('./')
from global_cfg      import *
from pyg_viewer      import *
from depth_img_proc  import *
from depth_cam_tools import *
from algo            import *

def draw_trace_mode(viewer,mode='vline',pos=0):
    if mode=='none':
        pass
    elif mode=='circle':
        viewer.draw_circle(960,215,30,line_wid=2)
        a=float(pos)*np.pi/180.0
        dx= math.cos(a)*30
        dy=-math.sin(a)*30
        viewer.draw_circle(int(960+dx),int(215+dy),5,line_wid=3)
        
    elif mode=='h-line':
        viewer.draw_hline((900,215),80,line_wid=3)
        pos=min(max(pos,-1.0),1.0)
        x=float(pos)*30.0+940
        viewer.draw_circle(int(x),215,5,line_wid=3)
        
    elif mode=='v-line':
        viewer.draw_vline((960,175),80,line_wid=3)
        pos=min(max(pos,-1.0),1.0)
        y=float(pos)*30.0+215
        viewer.draw_circle(960,int(y),5,line_wid=3)
        
    elif mode=='line-45':
        viewer.draw_line((310,10),(250,70),line_wid=2)
        pos=min(max(pos,-1.0),1.0)
        x=float(pos)*30.0+280
        y=-float(pos)*30.0+40
        viewer.draw_circle(int(x),int(y),5,line_wid=2)
        
    elif mode=='line-135':
        viewer.draw_line((250,10),(310,70),line_wid=2)
        pos=min(max(pos,-1.0),1.0)
        x=float(pos)*30.0+280
        y=float(pos)*30.0+40
        viewer.draw_circle(int(x),int(y),5,line_wid=2)

    elif mode=='point':
        viewer.draw_circle(960,205,5,line_wid=2)
        
    return


def mainloop_trk(algo, frame, viewer, bg=None):

    # cv2.imshow('img',bg)
    ####################
    # 点云旋转工具
    ####################
    if ENABLE_PC_ROTATE:
        dep_trans=depth_cam_trans_c(IMG_WID,IMG_HGT)
        dep_trans.cam_param_f(DEP_CAM_F)
        
        # 点云变换参数  
        T=pc_movz(-PC_ROT_CZ)                  # 点云变换矩阵
        T=np.dot(T,pc_rotx(PC_ROT_AX))
        T=np.dot(T,pc_movz(PC_ROT_CZ))
        
    ####################
    # 深度图预处理工具以及配置
    ####################
    dep_proc=depth_img_proc_c(init=False)
    dep_proc.set_param(img_wid=IMG_WID,img_hgt=IMG_HGT)
    
    dep_proc.dist_dmin=IMG_PROC_DMIN
    dep_proc.dist_dmax=IMG_PROC_DMAX
    
    dep_proc.dist_mask      =True   # 距离切割标志和门限
    dep_proc.amp_mask       =False  # 亮度切割标志和门限
    
    dep_proc.var_mask       =True   # 飞散点切割标志和门限
    dep_proc.var_th         =np.float32(0.01)
        
    dep_proc.fill_hole      =False  # 空洞填充
    
    dep_proc.median_filter  =True   # 空域中值滤波器  
    dep_proc.MEDIAN_FILTER_WIN=5
    
    dep_proc.gaussian_filter=False  # 空域高斯（低通）滤波器  
    dep_proc.iir_filter     =False  # 时域1阶IIR滤波器  
    dep_proc.wfir_filter    =False  # FIR滤波器
    dep_proc.cfir_filter    =False  # 复数FIR滤波器(注意，仅用于TOF相机，并且需要知道深度值和相位的转换方法，见对应class的成员变量scale_d2p)
    dep_proc.median3t_filter=False  # 时域中值滤波器
    dep_proc.gif_filter     =False  # 空域导向滤波器(注意：会导致距离失真)
    
    dep_proc.init_var()
    
    ####################
    # 主循环
    ####################

    img_amp=3e4-frame.copy().astype(np.float32) # FIXME!没有强度图，只能从深度图得到
    img_amp.shape=IMG_HGT,IMG_WID

    # 将深度图转换Z坐标
    img_dep=frame.copy().astype(np.float32)#*np.float32(DEP_CAM_SCALE)   # 注意，单位是M
    img_dep.shape=IMG_HGT,IMG_WID

    mask=np.ones_like(img_dep,dtype=bool)

    # 点云旋转变换
    # if ENABLE_PC_ROTATE:
        # 显示变换前的深度图
        # img_rgb=img_to_cmap(img_dep,mask,IMG_PROC_DMIN,1)
        # import dmcam
        # dist_cnt, img_rgb = dmcam.cmap_float(len(img_dep)*3,img_dep,dmcam.DMCAM_CMAP_OUTFMT_RGB,0.0,0.25)
        # img_rgb = cv2.resize(img_rgb, (640, 480))
        # viewer.update_pan_img_rgb(img_rgb,pan_id=(0,1))

        # # 将深度图转换成点云
        # pc=dep_trans.depth_to_pcloud(img_dep)
        #
        # # 点云变换，并将变换后的点云映射回深度图
        # pc_new=pc_trans(T,pc)
        # img_dep_new,mask_new=dep_trans.pcloud_to_depth(pc_new)
        #
        # img_dep=img_dep_new.copy().astype(np.float32)
        # mask=mask_new.copy()

    # 图像预处理
    img_dep_ori=img_dep.copy()
    img_dep,img_dep_u8,mask=dep_proc.calc(img_dep,img_amp,mask)
    # cv2.imshow("img_dep",mask.astype(np.uint8)*255)

    #############################################
    # 手指检测算法
    #############################################
    # 图像边界切割
    mask[:FTIP_TRK_Y_START,:]=False
    mask[FTIP_TRK_Y_END:,:]=False
    mask[:,:FTIP_TRK_X_START]=False
    mask[:,FTIP_TRK_X_END:]=False

    # xtip, ytip, q0, q1 = algo.calc_tip(img_dep, img_amp, mask=mask)

    (img_algo,mask_algo)=algo.calc(img_dep,img_amp,mask=mask)
    img_algo.shape=IMG_HGT,IMG_WID
    mask_algo.shape=IMG_HGT,IMG_WID

    # 显示算法结果
    img_rgb=img_to_cmap(img_algo,mask_algo,vmin=IMG_PROC_DMIN,vmax=IMG_PROC_DMAX)
    if bg is None:
        viewer.update_pan_img_rgb(img_rgb, pan_id=(0, 0))
    else:
        img_show = bg.copy()
        img_rgb[mask_algo==0]=[255,255,255]
        img_rgb=cv2.resize(img_rgb,(800,600))
        img_show[105:705,320:1120,:]=img_rgb
        viewer.update_pan_img_rgb(img_show,pan_id=(0,0))

    if np.sum(mask_algo.astype(int))==0:
        img_algo=np.zeros(IMG_SZ)
        mask_algo=np.ones(IMG_SZ,dtype=bool)

    mask_algo.shape=IMG_HGT,IMG_WID
    img_algo.shape =IMG_HGT,IMG_WID

    img_algo[mask_algo]=0

    # if algo.valid[algo.idx]==1:
    #     x,y=algo.xlist[algo.idx],algo.ylist[algo.idx]
    #     viewer.draw_circle(int(x),int(y),line_wid=2)

        # draw_trace_mode(viewer,mode=algo.trace_pattern['type'],pos=algo.trace_pattern['param'])
        # print('[%d] trace pattern:%s,param:%.2f'%(frame_cnt,algo.trace_pattern['type'],algo.trace_pattern['param']))

    return algo
    # # 刷新屏幕显示
    # viewer.update()
    # evt,param=viewer.poll_evt()


## 数手指
def mainloop_cntf():
    ENABLE_PC_ROTATE=False
    IMG_PROC_DMIN=0.0
    IMG_PROC_DMAX=0.8

    if USE_ZMQ:
        pass
        # context = zmq.Context()
        # subscriber = context.socket(zmq.SUB)
        # subscriber.connect(ZMQ_ADDR)
        # subscriber.setsockopt(zmq.SUBSCRIBE, b"DIST")
        # port = 7789

        # g_zmq_ctx = zmq.Context()
        # g_zmq_pub = g_zmq_ctx.socket(zmq.SUB)
        # g_zmq_pub.connect("tcp://127.0.0.1:%d" % port)
        # g_zmq_pub.setsockopt(zmq.SUBSCRIBE, b"DIST")  # All topics
        # g_zmq_pub.setsockopt(zmq.LINGER, 200)
    else:
        fname=PLAYBACK_FNAME+'_dep.bin'
        fp=open(fname,'rb')
        fp.seek(FRAME_BYTE_SZ*PLAYBACK_SKIP,0)

    ####################
    # 点云旋转工具
    ####################
    if ENABLE_PC_ROTATE:
        dep_trans=depth_cam_trans_c(IMG_WID,IMG_HGT)
        dep_trans.cam_param_f(DEP_CAM_F)
        
        # 点云变换参数  
        T=pc_movz(-PC_ROT_CZ)                  # 点云变换矩阵
        T=np.dot(T,pc_rotx(PC_ROT_AX))
        T=np.dot(T,pc_movz(PC_ROT_CZ))

    ####################
    # GUI显示工具
    ####################
    viewer=pyg_viewer_c(pan_wid=IMG_WID,pan_hgt=IMG_HGT,pan=(1,2))
        
    ####################
    # 深度图预处理工具以及配置
    ####################
    dep_proc=depth_img_proc_c(init=False)
    dep_proc.set_param(img_wid=IMG_WID,img_hgt=IMG_HGT)
    
    dep_proc.dist_dmin=IMG_PROC_DMIN
    dep_proc.dist_dmax=IMG_PROC_DMAX
    
    dep_proc.dist_mask      =True   # 距离切割标志和门限
    dep_proc.amp_mask       =False  # 亮度切割标志和门限
    
    dep_proc.var_mask       =True   # 飞散点切割标志和门限
    dep_proc.var_th         =np.float32(0.01)
        
    dep_proc.fill_hole      =False  # 空洞填充
    
    dep_proc.median_filter  =True   # 空域中值滤波器  
    dep_proc.MEDIAN_FILTER_WIN=5
    
    dep_proc.gaussian_filter=False  # 空域高斯（低通）滤波器  
    dep_proc.iir_filter     =False  # 时域1阶IIR滤波器  
    dep_proc.wfir_filter    =False  # FIR滤波器
    dep_proc.cfir_filter    =False  # 复数FIR滤波器(注意，仅用于TOF相机，并且需要知道深度值和相位的转换方法，见对应class的成员变量scale_d2p)
    dep_proc.median3t_filter=False  # 时域中值滤波器
    dep_proc.gif_filter     =False  # 空域导向滤波器(注意：会导致距离失真)
    
    dep_proc.init_var()
    
    ####################
    # 手指检测器
    ####################
    algo=multi_ftip_det_c()
    
    ####################
    # 主循环
    ####################
    frame_cnt=0
    while True:
        if USE_ZMQ: # 从ZMQ通道中读取一帧深度图
            pass
            # data = g_zmq_pub.recv()
            # while g_zmq_pub.getsockopt(zmq.RCVMORE):
            #     dist = g_zmq_pub.recv()
            #     data = np.fromstring(dist, dtype=np.float32)
            #     # data = data * 10000
            #     frame = data.reshape(240,320)
                # cv2.flip(frame, , frame)
            # address, contents = subscriber.recv_multipart()
            # w,h,d = struct.unpack('=lll', contents[0:12])
            # frame = np.frombuffer(contents[12:], np.uint16 if d == 2 else np.uint8).reshape(h,w)
        else:       # 从数据文件中读取一帧深度图
            frame=np.fromfile(fp,dtype=np.int16,count=IMG_SZ)
            if len(frame)<IMG_SZ:
                fp.seek(0, os.SEEK_SET)
                frame=np.fromfile(fp,dtype=np.int16,count=IMG_SZ)
        frame_cnt+=1
        cv2.flip(frame, 1, frame)
        # plt.imshow(frame)
        # plt.show()
        # cv2.imshow("frame",frame)
        # cv2.waitKey(1)
        
        img_amp=3e4-frame.copy().astype(np.float32) # FIXME!没有强度图，只能从深度图得到
        img_amp.shape=IMG_HGT,IMG_WID
        
        # 将深度图转换Z坐标
        img_dep=frame.copy().astype(np.float32)#*np.float32(DEP_CAM_SCALE)   # 注意，单位是M
        img_dep.shape=IMG_HGT,IMG_WID
        
        mask=np.ones_like(img_dep,dtype=bool)
        
        # 点云旋转变换
        if ENABLE_PC_ROTATE:
            # 显示变换前的深度图
            img_rgb=img_to_cmap(img_dep,mask,IMG_PROC_DMIN,IMG_PROC_DMAX)
            viewer.update_pan_img_rgb(img_rgb,pan_id=(0,1))
            
            # 将深度图转换成点云
            # pc=dep_trans.depth_to_pcloud(img_dep)
            #
            # # 点云变换，并将变换后的点云映射回深度图
            # pc_new=pc_trans(T,pc)
            # img_dep_new,mask_new=dep_trans.pcloud_to_depth(pc_new)
            #
            # img_dep=img_dep_new.copy().astype(np.float32)
            # mask=mask_new.copy()
            
        # 图像预处理
        img_dep_ori=img_dep.copy()
        img_dep,img_dep_u8,mask=dep_proc.calc(img_dep,img_amp,mask)
        
        #############################################
        # 手指检测算法
        #############################################
        # 图像边界切割
        mask[:FTIP_TRK_Y_START,:]=False
        mask[FTIP_TRK_Y_END:,:]=False
        mask[:,:FTIP_TRK_X_START]=False
        mask[:,FTIP_TRK_X_END:]=False
        
        (img_algo,mask_algo)=algo.calc(img_dep,img_amp,mask=mask)  
        img_algo.shape=IMG_HGT,IMG_WID
        mask_algo.shape=IMG_HGT,IMG_WID
        
        # 显示算法结果
        img_rgb=img_to_cmap(img_algo,mask_algo,vmin=IMG_PROC_DMIN,vmax=IMG_PROC_DMAX)
        finger_cnt=np.sum(np.array(algo.ytip_list)<algo.y_palm)
        if finger_cnt>0:
            cv2.putText(img_rgb,'%d'%finger_cnt,(200,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)

        
        viewer.update_pan_img_rgb(img_rgb,pan_id=(0,0))        
        
        
        viewer.update_pan_img_gray(algo.img_debug*255,pan_id=(0,1))
        #plt.imshow(algo.img_debug)
        #plt.show()
        
        if np.sum(mask_algo.astype(int))==0:
            img_algo=np.zeros(IMG_SZ)
            mask_algo=np.ones(IMG_SZ,dtype=bool)
        
        mask_algo.shape=IMG_HGT,IMG_WID
        img_algo.shape =IMG_HGT,IMG_WID
        
        img_algo[mask_algo]=0
        
        # 刷新屏幕显示
        viewer.update()
        evt,param=viewer.poll_evt()
        
        # 回放文件时，延迟等待
        if ENABLE_SIM and PLAYBACK_FPS>0:
            time.sleep(1.0/PLAYBACK_FPS)

    # 关闭ZMQ通道
    if USE_ZMQ:
        subscriber.close()
        context.term()

if __name__ == '__main__':
    mainloop_trk()
    
    # mainloop_cntf()
    
    # ENABLE_PC_ROTATE=False
    # mainloop_trk()
    

