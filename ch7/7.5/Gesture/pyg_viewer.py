#!/usr/bin/python3
# coding=utf-8

import sys
import cv2
import pygame
import numpy as np

sys.path.append('./')
from global_cfg import *

## pygame窗口显示管理
class pyg_viewer_c:
    def __init__(self,pan_wid=IMG_WID,pan_hgt=IMG_HGT,pan=(1,1),name='手势识别系统'):
        pygame.init()
        
        ## 板块尺寸
        self.pan_wid=pan_wid
        self.pan_hgt=pan_hgt
        self.pan_sz =(self.pan_wid,self.pan_hgt)    
        
        ## 窗口尺寸
        self.win_hgt=self.pan_hgt*pan[0]
        self.win_wid=self.pan_wid*pan[1]
                
        ## 二级图像（板块）缓存
        self.pan_surf={}
        px,py=np.meshgrid(range(pan[1]),range(pan[0]))
        self.pan_id_pos={}  # 存放每个板块的id和位置
        for x,y in zip(px.flatten(),py.flatten()):
            pan_id =(y,x)
            pan_pos=(self.pan_wid*x,self.pan_hgt*y)
            
            self.pan_id_pos[pan_id]=pan_pos
            self.pan_surf[pan_id] = pygame.Surface(self.pan_sz,0,24)
        
        ## 屏幕显示
        self.screen = pygame.display.set_mode((self.win_wid,self.win_hgt), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 24)
        pygame.display.set_caption(name)
        
        ## 存储鼠标状态
        self.mouse_button=[False,False,False]
        
        ## 定时器
        self.clock = pygame.time.Clock()

        return

        
    ## 将RGB图更新到对应板块
    def update_pan_img_rgb(self,img_rgb=[],pan_id=(0,0)):
        if len(img_rgb)==0: return False
        if not (pan_id in self.pan_id_pos): return False
        self.pan_surf[pan_id]=pygame.image.frombuffer(img_rgb,self.pan_sz,'RGB')
        return True

        
    ## 将灰度图更新到对应板块
    def update_pan_img_gray(self,img_gray=[],pan_id=(0,0)):
        if len(img_gray)==0: return False
        img_u8=img_gray.clip(0,255).astype(np.uint8).reshape((self.pan_hgt,self.pan_wid))
        img_rgb=np.dstack((img_u8,img_u8,img_u8))
        return self.update_pan_img_rgb(img_rgb,pan_id)


    ## 将灰度图转换成伪彩色，再更新到对应板块
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
        
        
    ## 屏幕更新
    def update(self,pan_id=(0,0)):
        for pan_id,pan_pos in self.pan_id_pos.items():
            self.screen.blit(self.pan_surf[pan_id],pan_pos)
            #print('pan_id:%d,%d, pan_pos:%d,%d'%(pan_id[0],pan_id[1],pan_pos[0],pan_pos[1]))
        pygame.display.update()
        pygame.display.flip()
        return

    ## 在给定的板块上画线
    def draw_line(self,start,end,pan_id=(0,0),color=pygame.color.THECOLORS["blue"],line_wid=8):
        if not (pan_id in self.pan_id_pos): return False
        s=self.pan_surf[pan_id]
        pygame.draw.line(s,color,start,end,line_wid)
        return True

    def draw_vline(self,start,hgt,pan_id=(0,0),color=pygame.color.THECOLORS["red"],line_wid=8):
        if start[1]>start[0]: return self.draw_vline((start[1],start[0]),hgt,pan_id,color,line_wid)
        
        end=(start[0],start[1]+hgt-1)
        return self.draw_line(start,end,pan_id=pan_id,color=color,line_wid=line_wid)
    
    def draw_hline(self,start,wid,pan_id=(0,0),color=pygame.color.THECOLORS["red"],line_wid=8):
        if start[1]>start[0]: return self.draw_hline((start[1],start[0]),wid,pan_id,color,line_wid)
        
        end=(start[0]+wid-1,start[1])
        return self.draw_line(start,end,pan_id=pan_id,color=color,line_wid=line_wid)

    ## 在给定的板块上画圆
    def draw_circle(self,x,y,r=10,pan_id=(0,0),color=pygame.color.THECOLORS["red"],line_wid=8):
        if not (pan_id in self.pan_id_pos): return False
        s=self.pan_surf[pan_id]
        if r>=line_wid:
            pygame.draw.circle(s, color,(x,y),r,line_wid)
            return True
        else:
            return False

    def put_text(self,text,x=110,y=1000,pan_id=(0,0),color=pygame.color.THECOLORS["green"]):
        if not (pan_id in self.pan_id_pos): return False
        s = self.pan_surf[pan_id]
        fontObj = pygame.font.SysFont('SimHei',32)  # 通过字体文件获得字体对象
        textSurfaceObj = fontObj.render(text, True, color)  # 配置要显示的文字
        textRectObj = textSurfaceObj.get_rect()  # 获得要显示的对象的rect
        textRectObj.center = (y, x)  # 设置显示对象的坐标
        s.blit(textSurfaceObj, textRectObj)  # 绘制字



    ## 退出
    def close(self): pygame.quit()
    
    ## 轮询用户事件
    def poll_evt(self,delay=0.0):
        self.sleep(delay)
        
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
    
    def sleep(self,delay):
        if delay>0: 
            self.clock.tick(delay)

    
## 单元测试
if __name__ == '__main__':
    pass
    
