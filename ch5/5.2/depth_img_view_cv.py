#!/usr/bin/python3
# -*- coding: utf-8 -*-

##############################
#        深度图显示工具         #
#         V20180801          #
#----------------------------#
# 20180801
# 1. 加入文件“虚拟”相机
# 2. 修改程序，直接使用实际相机和 “虚拟”相机
# 3. 测试使用直接dmcam的python接口
# 4. API名称修改
# 5. record_dep_amp函数入参减少
#
# 20180810
# 1. 为playback_cam_c加入skip_list功能
##############################

import sys

sys.path.append('./')
from config.global_cfg import *
from depth_cam_tools    import *
from cv_viewer          import *
from depth_img_proc_cv import *
img_hgt,img_wid=240,320
## 深度图回放相机
class playback_cam_c:
    def __init__(self,fname_dep=None,fname_amp=None,img_wid=320,img_hgt=240,skip=0,rewind=np.inf,skip_list=None):
        self.img_shape    = (img_hgt, img_wid)
        self.img_sz       = img_wid*img_hgt
        self.frame_byte_sz= self.img_sz*4
        self.skip_byte    = skip*self.frame_byte_sz
        self.rewind       = rewind
        self.skip_list    = skip_list   # {frame_cnt:skip, frame_cnt:skip, ...}
        self.skip         = skip

        self.frame_cnt    = self.skip
    
        if fname_dep is not None:
            print('[INF] open dep file:',fname_dep)
            self.fp_dep=open(fname_dep,'rb')
            self.fp_dep.seek(self.skip_byte,os.SEEK_SET)
        else:
            self.fp_dep=None
            
        if fname_amp is not None:
            print('[INF] open amp file:',fname_amp)
            self.fp_amp=open(fname_amp,'rb')
            self.fp_amp.seek(self.skip_byte,os.SEEK_SET)
        else:
            self.fp_amp=None
        return
    
    def get_dep_amp(self):
        if self.skip_list is not None:
            if self.frame_cnt in self.skip_list:
                if self.fp_dep is not None: self.fp_dep.seek(self.skip_list[self.frame_cnt]*self.frame_byte_sz,os.SEEK_CUR)
                if self.fp_amp is not None: self.fp_amp.seek(self.skip_list[self.frame_cnt]*self.frame_byte_sz,os.SEEK_CUR)
                self.frame_cnt+=self.skip_list[self.frame_cnt]

        if self.fp_dep is not None:
            frame_dep=np.fromfile(self.fp_dep,dtype=np.float32,count=self.img_sz)
            if (len(frame_dep)<self.img_sz) or (self.frame_cnt>self.rewind):
                self.fp_dep.seek(self.skip_byte,os.SEEK_SET)
                self.frame_cnt+=self.skip
                frame_dep=np.fromfile(self.fp_dep,dtype=np.float32,count=self.img_sz)
        else:
            frame_dep=None
        
        if self.fp_amp is not None:
            frame_amp=np.fromfile(self.fp_amp,dtype=np.float32,count=self.img_sz)
            if (len(frame_amp)<self.img_sz) or (self.frame_cnt>self.rewind):
                self.fp_amp.seek(self.skip_byte,os.SEEK_SET)
                if self.fp_dep is None: self.frame_cnt+=self.skip
                frame_amp=np.fromfile(self.fp_amp,dtype=np.float32,count=self.img_sz)
        else:
            frame_amp=None
        
        img_dep = None if frame_dep is None else frame_dep.reshape(self.img_shape)
        img_amp = None if frame_amp is None else frame_amp.reshape(self.img_shape)
        
        self.frame_cnt+=1
        return img_dep, img_amp, self.frame_cnt

    def close(self):
        if self.fp_dep is not None: 
            self.fp_dep.close()
            self.fp_dep=None

        if self.fp_amp is not None: 
            self.fp_amp.close()
            self.fp_amp=None
        return

    def poll_frame(self): return True


def playback_dep_amp(fname_dep=None, fname_amp=None,img_wid=320,img_hgt=240,dmin=0,dmax=5,fps=30):

    # 虚拟相机
    cam=playback_cam_c(fname_dep=fname_dep, fname_amp=fname_amp, img_wid=320, img_hgt=240)
    
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,2))
    
    while True:
        img_dep,img_amp,_=cam.get_dep_amp()
        if img_dep is not None: viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,0))
        if img_amp is not None: viewer.update_pan_img_gray(img_amp,pan_id=(0,1))

        # 刷新屏幕显示
        viewer.update()
        evt,_=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': break 
        time.sleep(1.0/float(fps))        
    return


def playback_dep_3d(fname,cam_f=640,cz=2.0,img_wid=320,img_hgt=240,dmin=0,dmax=5,fmt=np.float32,fps=30,plane_corr=None):

    delay=1.0/float(fps)
    img_sz=img_wid*img_hgt
    
    # 点云变换参数
    ax=ay=0.0                       # 点云旋转角度
    cz=cz                           # 点云旋转中心
    mz=0                            # 点云观察点位置
    T=np.eye(4,dtype=np.float32)    # 点云变换矩阵

    # 鼠标动作跟踪
    mouse_down=False
    mouse_x=mouse_y=0
    
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,2))
    
    # 深度图处理工具
    dep_trans=depth_cam_trans_c(img_wid=img_wid,img_hgt=img_hgt)
    dep_trans.cam_param_f(cam_f)  

    # 打开文件按
    fin=open(fname,'rb')
    while True:
        # 读取数据帧
        img_dep=np.fromfile(fin, dtype=fmt, count=img_sz)
        if len(img_dep)<img_sz: break
        
        # 简易平面校准
        if plane_corr is not None:
            img_dep=img_dep-plane_corr
            print(np.mean(img_dep))
        
        # 将深度图转换成点云
        pc=dep_trans.depth_to_pcloud(img_dep)
            
        # 点云变换，并将变换后的点云映射回深度图
        pc_new=pc_trans(T,pc)
        img_dep_new,mask=dep_trans.pcloud_to_depth(pc_new)
        
        # 将深度图转换成伪彩色，并更新对应的显示版面
        img_rgb=img_to_cmap(img_dep_new,mask,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3)
        viewer.update_pan_img_rgb(img_rgb,pan_id=(0,0))

        # 显示未旋转图像
        viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,1))

        # 刷新屏幕显示
        viewer.update()
        
        # 检查鼠标动作
        update_trans_mat=False
        evt,param=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': 
                break 
            elif evt=='md0': 
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
        time.sleep(delay)


def playback_dep(fname,img_wid=320,img_hgt=240,dmin=0,dmax=5,fmt=np.float32,fps=30):

    delay=1.0/float(fps)
    img_sz=img_wid*img_hgt
    
    fin=open(fname,'rb')
    
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,1))
    
    while True:
        img_dep=np.fromfile(fin, dtype=fmt, count=img_sz)
        if len(img_dep)<img_sz: break
        
        viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,0))
        
        # 刷新屏幕显示
        viewer.update()
        evt,_=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': 
                break 
        time.sleep(delay)        
    
    fin.close()


def play_dep_amp(cam,img_wid=320,img_hgt=240,dmin=0,dmax=5,fps=30):
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,2))
    
    frame_cnt=0
    while True:
        img_dep,img_amp,frame_cnt_new=cam.get_dep_amp()
        if frame_cnt_new==frame_cnt:
            time.sleep(1.0/float(fps))
            continue
        frame_cnt=frame_cnt_new
        
        if img_dep is not None: viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,0))
        if img_amp is not None: viewer.update_pan_img_gray(img_amp,pan_id=(0,1))

        # 刷新屏幕显示
        viewer.update()
        evt,_=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': break 
    return
    

def play_dep_3d(cam,cam_f=640,cz=2.0,img_wid=320,img_hgt=240,dmin=0,dmax=5):
    # 点云变换参数
    ax=ay=0.0                       # 点云旋转角度
    cz=cz                           # 点云旋转中心
    mz=0                            # 点云观察点位置
    T=np.eye(4,dtype=np.float32)    # 点云变换矩阵

    # 鼠标动作跟踪
    mouse_down=False
    mouse_x=mouse_y=0
    
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,2))
    
    # 深度图处理工具
    dep_trans=depth_cam_trans_c(img_wid=img_wid,img_hgt=img_hgt)
    dep_trans.cam_param_f(cam_f)  

    while True:
        # 读取数据帧
        img_dep,img_amp,frame_id=cam.get_dep_amp()
        
        # 检查是否有新的数据帧到达
        if img_dep is None or img_amp is None:
            time.sleep(0.001)
            continue
        elif frame_id%10==0: 
            print(frame_id)
        
        # 将深度图转换成点云
        pc=dep_trans.depth_to_pcloud(img_dep)
            
        # 点云变换，并将变换后的点云映射回深度图
        pc_new=pc_trans(T,pc)
        img_dep_new,mask=dep_trans.pcloud_to_depth(pc_new)
        
        # 将深度图转换成伪彩色，并更新对应的显示版面
        img_rgb=img_to_cmap(img_dep_new,mask,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3)
        viewer.update_pan_img_rgb(img_rgb,pan_id=(0,0))

        # 显示未旋转图像
        viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,1))

        # 刷新屏幕显示
        viewer.update()
        
        # 检查鼠标动作
        update_trans_mat=False
        evt,param=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': 
                break 
            elif evt=='md0': 
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


def play_dep(cam,img_wid=320,img_hgt=240,dmin=0,dmax=5):
    # GUI显示
    viewer=cv_viewer_c(pan_wid=img_wid,pan_hgt=img_hgt,pan_num=(1,1))
    
    frame_id=0
    while True:
        # 读取数据帧
        img_dep,_,frame_id_new=cam.get_dep_amp()
        
        # 检查是否有新的数据帧到达
        if frame_id_new==frame_id:
            time.sleep(0.001)
            continue
        else:
            frame_id=frame_id_new
            if frame_id%10==0: print(frame_id)
        
        viewer.update_pan_img_rgb(img_to_cmap(img_dep,None,vmin=dmin,vmax=dmax).reshape(img_hgt,img_wid,3),pan_id=(0,0))

        # 刷新屏幕显示
        viewer.update()
        evt,_=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': 
                break       
    return


## 采集dep和amp数据的程序
# 输入
#   fname_dep       保存的深度图文件名, None表示不保存
#   fname_amp       保存的强度图文件名, None表示不保存
#   num             保存的数据帧数
#   flipud/fliplr   图像反转标志
#   auto_save       是否自动保存
#                   True —— 一次采集num帧数据，然后保存，
#                   False—— 等待用户按键q退出，保存推出前最后采集的最多num帧
#   cam             象机设备对象
def record_dep_amp(cam,
                   fname_dep=None, fname_amp=None, \
                   num=100,\
                   flipud=False,fliplr=False,\
                   auto_save=True):
    
    viewer=cv_viewer_c(pan_wid=320, pan_hgt=240, pan_num=(1,2))  # 开启显示
    
    frames_dep=[]
    frames_amp=[]
    frame_id=0
    while True:
        img_dep,img_amp,frame_id_new=cam.get_dep_amp()
        
        # 检查是否有新的数据帧到达
        if frame_id==frame_id_new:
            time.sleep(0.001)
            continue
        else:
            frame_id=frame_id_new
            if frame_id%10==0:  print(frame_id)
        
        # 保存数据
        if fname_dep is not None: 
            frames_dep.append(img_dep.copy().flatten())
            if len(frames_dep)>num: frames_dep.pop(0)
        if fname_amp is not None:
            frames_amp.append(img_amp.copy().flatten())
            if len(frames_amp)>num: frames_amp.pop(0)
            
        if auto_save:
            if (len(frames_dep)==num) or (len(frames_amp)==num): break
        
        # 显示图像
        viewer.update_pan_img_rgb(img_to_cmap(img_dep,mask=None,vmin=0,vmax=5).reshape(img_hgt,img_wid,3),pan_id=(0,0))
        viewer.update_pan_img_gray(img_amp*0.1,pan_id=(0,1))
        
        # 屏幕刷新
        viewer.update()
        evt,_=viewer.poll_evt()
        if evt is not None: 
            if evt=='quit': break 
    
    # 保存采集数据
    if fname_dep is not None: 
        print('saving depth data to %s (%d frames)'%(fname_dep,len(frames_dep)))
        fd=open(fname_dep,'wb')
        for n in range(len(frames_dep)): frames_dep[n].astype(np.float32).tofile(fd)   
        fd.close()
                
    if fname_amp is not None: 
        print('saving amplitude data to %s (%d frames)'%(fname_amp,len(frames_amp)))
        fd=open(fname_amp,'wb')
        for n in range(len(frames_amp)): frames_amp[n].astype(np.float32).tofile(fd)   
        fd.close()


if __name__ == '__main__':
    #from server_comm import *
    #cam=tof_cam_shm_c(flipud=False,fliplr=False)# 开启基于共享内存通信的TOF相机
    
    ## 实时显示深度图和灰度图
    if False:
        from dmcam_dev import *
        cam=dmcam_dev_c()
        play_dep_amp(cam)

    ## 实时显示深度图
    if False:
        from dmcam_dev import *
        cam=dmcam_dev_c()
        play_dep(cam)
    
    ## 实时显示3D点云和深度图
    if False:
        from dmcam_dev import *
        cam=dmcam_dev_c()
        play_dep_3d(cam,cz=2.5)

    ## 回放深度图和灰度图
    if False:
        playback_dep_amp('./data/dep.bin','./data/amp.bin') 

    ## 回放深度图
    if False:
        playback_dep('./data/dep.bin') 

    ## 回放深度图和灰度图
    if True:
        playback_dep_3d('./data/dep.bin') 
    
    ## 财经保存深度图和灰度图
    if False:
        from dmcam_dev import *
        cam=dmcam_dev_c()
        record_dep_amp(cam,'./data/dep.bin','./data/amp.bin') 
