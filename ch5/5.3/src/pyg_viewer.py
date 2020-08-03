#!/usr/bin/python3
# coding=utf-8

import sys
import cv2
import pygame
import os
#os.environ["SDL_VIDEODRIVER"] = 'windib'
import numpy as np

sys.path.append('./')
from global_cfg import *

# pygame窗口显示管理
class pyg_viewer_c:
    def __init__(self,pan_wid=IMG_WID,pan_hgt=IMG_HGT,pan=(1,1),name='pygame window'):
        pygame.init()
        
        # 板块尺寸
        self.pan_wid=pan_wid
        self.pan_hgt=pan_hgt
        self.pan_sz =(self.pan_wid,self.pan_hgt)    
        
        # 窗口尺寸
        self.win_wid=self.pan_wid*pan[1]
        self.win_hgt=self.pan_hgt*pan[0]
                
        # 二级图像（板块）缓存
        self.pan_surf={}
        px,py=np.meshgrid(range(pan[1]),range(pan[0]))
        self.pan_id_pos={}  # 存放每个板块的id和位置
        for x,y in zip(px.flatten(),py.flatten()):
            pan_id =(x,y)
            pan_pos=(self.pan_wid*x,self.pan_hgt*y)
            
            self.pan_id_pos[pan_id]=pan_pos
            self.pan_surf[pan_id] = pygame.Surface(self.pan_sz,0,24)
        
        # 屏幕显示
        self.screen = pygame.display.set_mode((self.win_wid,self.win_hgt), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 24)
        pygame.display.set_caption(name)
        
        # 存储鼠标状态
        self.mouse_button=[False,False,False]
        
        return

        
    # 将RGB图更新到对应板块
    def update_pan_img_rgb(self,img_rgb=[],pan_id=(0,0)):
        if len(img_rgb)==0: return False
        if not (pan_id in self.pan_id_pos): return False
        self.pan_surf[pan_id]=pygame.image.frombuffer(img_rgb,self.pan_sz,'RGB')
        return True

        
    # 将灰度图更新到对应板块
    def update_pan_img_gray(self,img_gray=[],pan_id=(0,0)):
        if len(img_gray)==0: return False
        img_u8=img_gray.clip(0,255).astype(np.uint8).reshape((self.pan_hgt,self.pan_wid))
        img_rgb=np.dstack((img_u8,img_u8,img_u8))
        return self.update_pan_img_rgb(img_rgb,pan_id)


    # 将灰度图转换成伪彩色，再更新到对应板块
    def update_pan_img_gray_cmap(self,img_gray=[],mask=[],pan_id=(0,0)):
        if len(img_gray)==0: return False
        
        if len(mask)>0:
            vmin=np.min(img_gray[mask])
            vmax=np.max(img_gray[mask])
        else:
            vmin=np.min(img_gray)
            vmax=np.max(img_gray)
            
        img_norm=np.float32(img_gray-vmin)/np.float32(vmax-vmin+1)
        img_u8=np.uint8(img_norm*255)
        img_rgb=cv2.applyColorMap(img_u8,CV_CMAP_COLOR)
        if len(mask)>0:
            img_rgb[~mask,:]=0
        self.update_pan_img_rgb(img_rgb,pan_id)
        return
        
        
    # 屏幕更新
    def update(self,pan_id=(0,0)):
        for pan_id,pan_pos in self.pan_id_pos.items():
            self.screen.blit(self.pan_surf[pan_id],pan_pos)
            #print('pan_id:%d,%d, pan_pos:%d,%d'%(pan_id[0],pan_id[1],pan_pos[0],pan_pos[1]))
        pygame.display.update()
        pygame.display.flip()
        return

    # 在给定的板块上画线
    def draw_line(self,start,end,pan_id=(0,0),color=pygame.color.THECOLORS["white"],line_wid=8):
        if not (pan_id in self.pan_id_pos): return False
        s=self.pan_surf[pan_id]
        pygame.draw.line(s,color,start,end,line_wid)
        return True


    # 在给定的板块上画圆
    def draw_circle(self,x,y,r=10,pan_id=(0,0),color=pygame.color.THECOLORS["white"],line_wid=8):
        if not (pan_id in self.pan_id_pos): return False
        s=self.pan_surf[pan_id]
        if r>=line_wid:
            pygame.draw.circle(s, color,(x,y),r,line_wid)
            return True
        else:
            return False

    ## 退出
    def close(self): pygame.quit()
    
    ## 轮询用户事件
    def poll_evt(self):
        for event in pygame.event.get():    # User did something
            if event.type == pygame.QUIT:   # If user clicked close
                return 'quit',''

            elif event.type == pygame.KEYDOWN:
                if event.key == 27 or event.key == ord('q'):  
                    return 'quit',''
                elif event.key == ord('s'):
                    return 'key','s'
                else:
                    return 'kd',event.key
                    
            elif event.type ==  pygame.MOUSEBUTTONDOWN:
                pressed_array = pygame.mouse.get_pressed()
                if pressed_array[0]==1: 
                    self.mouse_button[0]=True
                    return 'md0', pygame.mouse.get_pos()
                if pressed_array[1]==1: 
                    self.mouse_button[1]=True
                    return 'md1', pygame.mouse.get_pos()
                if pressed_array[2]==1: 
                    self.mouse_button[2]=True
                    return 'md2', pygame.mouse.get_pos()
                if event.button==5:     
                    return 'mw0', pygame.mouse.get_pos()
                if event.button==4:     
                    return 'mw1', pygame.mouse.get_pos()
                
            elif event.type ==  pygame.MOUSEBUTTONUP:
                pressed_array = pygame.mouse.get_pressed()
                if pressed_array[0]==0 and self.mouse_button[0]==True:
                    self.mouse_button[0]=False 
                    return 'mu0', pygame.mouse.get_pos()
                if pressed_array[1]==0 and self.mouse_button[1]==True:
                    self.mouse_button[1]=False
                    return 'mu1', pygame.mouse.get_pos()
                if pressed_array[2]==0 and self.mouse_button[2]==True:
                    self.mouse_button[2]=False
                    return 'mu2', pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEMOTION:
                pos = pygame.mouse.get_pos()
                x,y = pos[0], pos[1]
                return 'mm',pygame.mouse.get_pos()
                
            else:
                return None,''
        return None,''


# 图像打标工具编辑器
class img_tagging_tool_c:
    def __init__(self,img=None,color_mode='gray',tag_color=[0,255]):
        self.viewer=pyg_viewer_c()
        
        self.scale    =1.0
        self.scale_min=1.0
        self.scale_max=40.0
        self.scale_step=1.1
        
        self.color_mode=color_mode
        self.tag_color=tag_color
        
        if img is None:
            if color_mode=='gray':
                self.img=np.zeros((IMG_HGT,IMG_WID))
            else:
                self.img=np.zeros((IMG_HGT,IMG_WID,3))
        self.set_img(img)
        
        return


    ## 计算缩放图
    def calc_img_scale(self):
        img_tag=self.img.copy()
        
        num=len(self.tag_color)
        if self.color_mode=='gray':
            for k in range(num): img_tag[self.tag==k]=self.tag_color[k]
            self.img_scale=img_tag[self.ymap.flatten(),self.xmap.flatten()].reshape(IMG_HGT,IMG_WID).copy()
        elif self.color_mode=='rgb':
            for k in range(num): img_tag[self.tag==k,:]=self.tag_color[k]
            self.img_scale=np.dstack((img_tag[self.ymap.flatten(),self.xmap.flatten(),0].reshape(IMG_HGT,IMG_WID).copy(),
                                      img_tag[self.ymap.flatten(),self.xmap.flatten(),1].reshape(IMG_HGT,IMG_WID).copy(),
                                      img_tag[self.ymap.flatten(),self.xmap.flatten(),2].reshape(IMG_HGT,IMG_WID).copy()))
        return


    # 计算缩放图并更新屏幕图像显示
    def update_img(self):
        self.calc_img_scale()
        if   self.color_mode=='gray': self.viewer.update_pan_img_gray(self.img_scale)
        elif self.color_mode=='rgb' : self.viewer.update_pan_img_rgb (self.img_scale)
        self.viewer.update()
        return
        
    
    # 设置被编辑图像
    def set_img(self,img):
        self.img=img.copy()
        self.tag=-np.ones((IMG_HGT,IMG_WID)).astype(np.int16)
        self.reset_scale()
        self.update_img()
        return


    # 尺度复位到1.0
    def reset_scale(self):
        self.scale=1.0
        x=np.arange(IMG_WID)
        y=np.arange(IMG_HGT)
        self.xmap,self.ymap=np.meshgrid(x,y)
        return


    def img_scaling(self,zoom=0):
        if   zoom>0: self.scale*=self.scale_step
        elif zoom<0: self.scale/=self.scale_step
        else:        return
            
        self.scale=min(40.0,self.scale)
        self.scale=max(1.0 ,self.scale)
        
        # 缩放图中像素在原图中的像素坐标
        x=np.clip(np.round(np.arange(IMG_WID).astype(np.float32)/self.scale).astype(np.int),0,IMG_WID-1)
        y=np.clip(np.round(np.arange(IMG_HGT).astype(np.float32)/self.scale).astype(np.int),0,IMG_HGT-1)
        self.xmap,self.ymap=np.meshgrid(x,y)        
        return
    

    def scale_down(self): self.img_scaling(-1)
    def scale_up  (self): self.img_scaling(+1)

    
    ## 编辑图像
    def edit(self):
        import time
        while True:
            evt,param=self.viewer.poll_evt()
            if   evt=='mw0': 
                self.scale_down()   # 尺度变换
                self.update_img()
            elif evt=='mw1': 
                self.scale_up()
                self.update_img()
            elif evt=='md2': 
                self.reset_scale()
            elif evt=='md0': # 像素编辑
                x,y=param[0],param[1]
                xm,ym=self.xmap[y,x],self.ymap[y,x]
                
                self.tag[ym,xm]+=1
                if self.tag[ym,xm]>=len(self.tag_color):
                    self.tag[ym,xm]=-1
                
                self.update_img()
            elif evt=='quit': break
            
            time.sleep(0.01)


# 图像打标工具编辑器(黑白图)
def test1():
    img_gray=(np.random.rand(IMG_HGT,IMG_WID)*255.0).astype(np.uint8)
    dut=img_tagging_tool_c(img_gray,
                           color_mode='gray',
                           tag_color=[0,255])
    dut.edit()
    return


# 图像打标工具编辑器(彩色图)
def test2():
    img_rgb=(np.random.rand(IMG_HGT,IMG_WID,3)*255.0).astype(np.uint8)
    dut=img_tagging_tool_c(img_rgb,
                           color_mode='rgb',
                           tag_color=[np.array([  0,  0,  0]),
                                      np.array([255,  0,  0]),
                                      np.array([  0,255,  0]),
                                      np.array([  0,  0,255]),
                                      np.array([255,255,255])])
    dut.edit()

    
## 单元测试
if __name__ == '__main__':
    test2()
    
