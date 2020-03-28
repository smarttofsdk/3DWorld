#!/usr/bin/python3
# coding=utf-8

import sys
import cv2
import numpy as np

sys.path.append('./')

##############################
#     基于OpenCV的显示工具     #
#         V20181108          #
#----------------------------#
# 20181107
# 实现cv_viewer的鼠标拖动功能
# 20181108
# 实现cv_viewer的窗口关闭功能
##############################


## 基于opencv的cv窗口显示管理
class cv_viewer_c:
    def __init__(self,pan_hgt=240,pan_wid=320,pan_num=(1,1),name='cv window'):
        ## 窗口
        self.name=name
        cv2.namedWindow(self.name)   
        
        ## 板块尺寸
        self.pan_wid=pan_wid
        self.pan_hgt=pan_hgt
        self.pan_sz =(self.pan_wid,self.pan_hgt)    
        self.pan_num_y,self.pan_num_x=pan_num
        
        ## 窗口尺寸
        self.win_hgt=self.pan_hgt*self.pan_num_y
        self.win_wid=self.pan_wid*self.pan_num_x

        ## 二级图像（板块）缓存
        self.img_show=np.zeros((self.win_hgt,self.win_wid,3),dtype=np.uint8)

        ## 鼠标
        self.mouse_evt=None
        self.mouse_pos=0,0
        cv2.setMouseCallback(name,self.mouse_callback)
        
        return


    ## 将RGB图更新到对应板块
    def update_pan_img_rgb(self,img_rgb=None,pan_id=(0,0)):
        if img_rgb is None: return False
        
        pan_id_y,pan_id_x=pan_id
        if pan_id_y<0 or pan_id_y>=self.pan_num_y or \
           pan_id_x<0 or pan_id_x>=self.pan_num_x:
           return False
        
        
        xs=pan_id_x*self.pan_wid
        ys=pan_id_y*self.pan_hgt
        xe=xs+self.pan_wid
        ye=ys+self.pan_hgt
        
        self.img_show[ys:ye,xs:xe,0]=img_rgb[0:self.pan_hgt,0:self.pan_wid,2]
        self.img_show[ys:ye,xs:xe,1]=img_rgb[0:self.pan_hgt,0:self.pan_wid,1]
        self.img_show[ys:ye,xs:xe,2]=img_rgb[0:self.pan_hgt,0:self.pan_wid,0]
        return True
        
        
    ## 将灰度图更新到对应板块
    def update_pan_img_gray(self,img_gray=None,pan_id=(0,0)):
        if img_gray is None: return False
        img_gray[img_gray>255]=255
        pan_id_y,pan_id_x=pan_id
        if pan_id_y<0 or pan_id_y>=self.pan_num_y or \
           pan_id_x<0 or pan_id_x>=self.pan_num_x:
           return False
        
        xs=pan_id_x*self.pan_wid
        ys=pan_id_y*self.pan_hgt
        xe=xs+self.pan_wid
        ye=ys+self.pan_hgt
        
        self.img_show[ys:ye,xs:xe,0]=img_gray[0:self.pan_hgt,0:self.pan_wid]
        self.img_show[ys:ye,xs:xe,1]=img_gray[0:self.pan_hgt,0:self.pan_wid]
        self.img_show[ys:ye,xs:xe,2]=img_gray[0:self.pan_hgt,0:self.pan_wid]
        return True
        
        
    ## 屏幕更新
    def update(self): cv2.imshow(self.name, self.img_show)

    ## 在给定的板块上画线
    def draw_line(self,xs,ys,xe,ye,pan_id=(0,0),color=(255,255,255),line_wid=5):
        pan_id_y,pan_id_x=pan_id
        if pan_id_y<0 or pan_id_y>=self.pan_num_y or \
           pan_id_x<0 or pan_id_x>=self.pan_num_x:
           return False
        
        x0=pan_id_x*self.pan_wid
        y0=pan_id_y*self.pan_hgt
        
        cv2.line(self.img_show,(x0+xs,y0+ys),(x0+xe,y0+ye),color,line_wid)
        return True


    ## 在给定的板块上画圆
    def draw_circle(self,x,y,r=10,pan_id=(0,0),color=(255,255,255),line_wid=5):
        pan_id_y,pan_id_x=pan_id
        if pan_id_y<0 or pan_id_y>=self.pan_num_y or \
           pan_id_x<0 or pan_id_x>=self.pan_num_x:
           return False
        
        x0=pan_id_x*self.pan_wid
        y0=pan_id_y*self.pan_hgt
        
        cv2.circle(self.img_show, (x0+x, y0+y), r, color, line_wid)
        return True
    
    
    def draw_text(self,s,x,y,pan_id=(0,0),color=(255,255,255),font_size=0.8, line_wid=2):
        pan_id_y,pan_id_x=pan_id
        x0=pan_id_x*self.pan_wid
        y0=pan_id_y*self.pan_hgt
        cv2.putText(self.img_show,s,(x0+x,y0+y),cv2.FONT_HERSHEY_SIMPLEX,font_size,color,line_wid)
        return


    ## 退出
    def close(self): pass
    
    
    ## 轮询用户事件
    def poll_evt(self,delay=1):
        self.update()
        key=cv2.waitKey(delay)&0xFF
        if key==ord('q'): return 'quit',None
        if key==ord('s'): return 'save',None
        
        if cv2.getWindowProperty(self.name,1) < 1:        
            return 'quit',None
            
        evt_to_str={cv2.EVENT_LBUTTONDOWN:'md0', cv2.EVENT_LBUTTONUP:'mu0', \
                    cv2.EVENT_MBUTTONDOWN:'md1', cv2.EVENT_MBUTTONUP:'mu1', \
                    cv2.EVENT_RBUTTONDOWN:'md2', cv2.EVENT_RBUTTONUP:'mu2', \
                    cv2.EVENT_MOUSEMOVE:'mm'}
        
        if self.mouse_evt in evt_to_str:
            evt=evt_to_str[self.mouse_evt]
            self.mouse_evt=None
            return evt,self.mouse_pos
        return None, None
    

    def mouse_callback(self,evt,x,y,flags,param):
        if evt in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, \
                   cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONUP, \
                   cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP]:
            self.mouse_evt=evt
        elif evt == cv2.EVENT_MOUSEMOVE:
            self.mouse_evt=evt
            self.mouse_pos=x,y

    

## 单元测试
if __name__ == '__main__':
    viewer=cv_viewer_c(pan_hgt=240,pan_wid=320,pan_num=(2,2))
    for k in range(500):
        img00=np.random.randint(0,256,(240,320,3))
        img01=np.random.randint(0,256,(240,320))
        
        img10=np.ones((240,320,3))
        img10[:,:,0]=img10[:,:,0]*abs(((k*3)%400)-200)
        img10[:,:,1]=img10[:,:,1]*abs(((k*4)%400)-200)
        img10[:,:,2]=img10[:,:,2]*abs(((k*5)%400)-200)
        
        img11=img10.copy()
        
        viewer.update_pan_img_rgb (img00,(0,0))
        viewer.update_pan_img_gray(img01,(0,1))
        viewer.update_pan_img_rgb (img10,(1,0))
        viewer.update_pan_img_rgb (img11,(1,1))
        
        viewer.draw_line  (k%320, (k*2+10)%240, (k**2-k)%320, (k**3-k*23)%240, pan_id=(1,1))
        viewer.draw_circle((k+20)%320, (k*5+20)%240, 10, pan_id=(1,1))
        viewer.draw_text  ('%d'%k,(k+20)%320, (k*5+20)%240, pan_id=(1,0))
        
        evt,_=viewer.poll_evt()
        if evt=='quit': break
        if k%20==0: print(k)
    viewer.close()
        
    
