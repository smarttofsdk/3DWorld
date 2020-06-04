# -*- coding: utf-8 -*-

from depth_cam_tools import *

import numpy as np
import cv2
import os
import zipfile


## 转换图片分辨率
# Kinect v2相机参数
dep_trans = depth_cam_trans_c(512, 424)
dep_trans.cam_param_k(1.0 / 367.2420124)

# tof相机参数
tof_trans = depth_cam_trans_c(320, 240)
tof_trans.cam_param_k(1.0 / 329.69729)
# tof_trans.cam_param_k(1.0 / 260.64507)

# dep_trans.cam_mov(0, -1.7, -2.5)
# tof_trans.cam_mov(0, -1.7, -2.5)
def change_res(img_dep):
    pc = dep_trans.depth_to_pcloud(img_dep / 1000.0)
    img_dep_new, mask = tof_trans.pcloud_to_depth(pc)
    img_dep_new[~mask] = 0
    return img_dep_new * 1000


def subdir(path, namelist):
    st = namelist.index(path) + 1
    end = st
    ln = len(namelist)
    while end < ln and namelist[end][-1] != '/':
        end += 1
    namelist = namelist[st:end]
    namelist.sort()
    return namelist


class ImShowMouse:
    """
    # 显示鼠标处的图像值
    """
    def __init__(self):
        self.mx,self.my,self.mflag = 0,0,1

    def getxy(self):
        return self.mx, self.my

    def get_mousexy(self, event, x, y, flags, param):
        if self.mflag and event == cv2.EVENT_MOUSEMOVE:
            self.mx, self.my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mflag ^= 1
            print(x,y)

    def imshow(self, windowname, img, clo=(255, 255, 255)):
        imgc = img.copy()
        cv2.namedWindow(windowname)
        cv2.setMouseCallback(windowname, self.get_mousexy)

        self.my, self.mx = min(self.my, imgc.shape[0] - 1), min(self.mx, imgc.shape[1] - 1)
        cv2.circle(imgc,(self.mx, self.my),1,clo,2)
        cv2.putText(imgc, f'{img[self.my, self.mx]}', (self.mx, self.my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clo, 2, cv2.LINE_AA)
        cv2.imshow(windowname, imgc)

cvm = ImShowMouse()

# 保存字典
def save_dict(filename, dict_name):
    with open(filename, 'w') as f:
        f.write(str(dict_name))

# 读取字典
def read_dict(filename):
    with open(filename, 'r') as f:
        a = f.read()
        dict_name = eval(a)
        return dict_name


## 保存视频
def write_avi(frame_array, fname='video.avi', fps=15, frame_hgt=240, frame_wid=320):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(fname,
                            fourcc,
                            fps,
                            (frame_wid, frame_hgt))

    num = len(frame_array)
    print('writing %d frames into video file %s' % (num, fname))
    for frame in frame_array:
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame_RGB)

    print('end of writing')
    video.release()


## 保存动态GIF
def create_gif(filename, gifframes, dur=0.05):
    import imageio
    imageio.mimsave(filename, gifframes, 'GIF', duration=dur)
    return



## 从骨架文件中读取关节点数据
def read_skeleton_file(filename, mode='0'):
    # mode :'0'  NTU数据集
    # 'j'  自采Kinect数据
    # 'diedao' --人工标注的tof数据
    if mode == 'diedao':
        with open(filename, 'r') as fileid:
            bodyinfo = {}
            framecount = fileid.readline()
            while framecount:
                framecount = int(framecount)
                bodycount = int(fileid.readline())
                bodys = []
                for b in range(bodycount):
                    jointCount = 15
                    joints = {}
                    jointinfo = list(map(float, fileid.readline().split()))
                    for j in range(jointCount):
                        joint = [jointinfo[2*j+1], jointinfo[2*j+2]]
                        if joint[0] != -1:
                            joints[j+1] = joint
                    bodys.append(joints)
                bodyinfo[framecount] = bodys
                framecount =fileid.readline()
        return bodyinfo

    else:  # mode == '0' or 'j'
        fileid = open(filename, 'r')
        framecount = int(fileid.readline())  # no of the recorded frames
        bodyinfo = {}

        for f in range(framecount):
            bodycount = int(fileid.readline())
            bodys = []
            for b in range(bodycount):
                fileid.readline()  # skip one line
                jointCount = int(fileid.readline())  # no of joints (25)
                joints = {}
                for j in range(jointCount):
                    jointinfo = list(map(float, fileid.readline().split()))
                    jointinfo.insert(0, 0.0)
                    joint = {}

                    # # 3D location of the joint j
                    # joint['x'] = jointinfo[1]
                    # joint['y'] = jointinfo[2]
                    # joint['z'] = jointinfo[3]

                    # 2D location of the joint j in corresponding depth/IR frame
                    if mode == 'j':
                        joint = [jointinfo[1], jointinfo[2]]
                    else:
                        joint = [jointinfo[4], jointinfo[5]]

                    # # 2D location of the joint j in corresponding RGB frame
                    # joint['colorX'] = jointinfo[6]
                    # joint['colorY'] = jointinfo[7]
                    #
                    # # The orientation of the joint j
                    # joint['orientationW'] = jointinfo[8]
                    # joint['orientationX'] = jointinfo[9]
                    # joint['orientationY'] = jointinfo[10]
                    # joint['orientationZ'] = jointinfo[11]
                    #
                    # # The tracking state of the joint j
                    # joint['trackingState'] = jointinfo[12]

                    if joint[0]!=-1:
                        joints[j+1] = joint
                bodys.append(joints)
            bodyinfo[f] = bodys
        fileid.close()
        return bodyinfo


class FolderImage:
    def __init__(self, folderName, rootPath='', skelFile='', skelMode ='0', depSuf='/dep', irSuf='/ir', bgSuf='/bg'):
        folderName = folderName.strip('/')
        rootPath = rootPath.strip('/')+'/'
        self.depSuf = depSuf.rstrip('/')+'/' if depSuf  else depSuf
        self.irSuf = irSuf.rstrip('/')+'/' if irSuf else irSuf
        self.bgSuf = bgSuf.rstrip('/') + '/' if bgSuf else bgSuf
        self.skeletonFileName = skelFile

        # 配置dep文件夹
        self.img_id = -1
        self.currentfolder = rootPath + folderName

        depthfolder = self.currentfolder+self.depSuf
        self.dep_fn = [f'{depthfolder}{fn}'for fn in os.listdir(depthfolder)]
        self.dep_name = os.listdir(depthfolder)
        self.len_fn = len(self.dep_fn)

        # 配置ir文件夹
        if self.irSuf:
            irfolder = self.currentfolder+self.irSuf
            if os.path.exists(irfolder):
                self.ir_fn = [f'{irfolder}{fn}'for fn in os.listdir(irfolder)]
            else:
                self.irSuf = ''

        # 配置bg文件夹
        if self.bgSuf:
            bgfolder = self.currentfolder + self.bgSuf
            if os.path.exists(bgfolder):
                self.bg_fn = [f'{bgfolder}{fn}' for fn in os.listdir(bgfolder)]
            else:
                self.bgSuf = ''

        # 配置骨架文件
        if self.skeletonFileName:
            self.skeletonFileName = self.skeletonFileName.format(folderName)
            if os.path.exists(self.skeletonFileName):
                self.bodyinfo = read_skeleton_file(self.skeletonFileName, mode=skelMode)
            else:
                self.skeletonFileName = ''

    # 下一张图片，返回是否成功
    def nextImage(self):
        self.img_id = max(0, self.img_id+1)
        return self.img_id < self.len_fn

    # 获取当前图片
    def get_depImage(self):
        return cv2.imread(self.dep_fn[self.img_id], -1).astype('float32')

    def get_irImage(self):
        if self.irSuf:
            return cv2.imread(self.ir_fn[self.img_id], -1).astype('float32')
        else:
            return None

    def get_bgImage(self, id):
        if self.bgSuf:
            return cv2.imread(self.bg_fn[id-1], -1).astype('float32')
        else:
            return None

    def get_skeletons(self):
        if self.skeletonFileName:
            return self.bodyinfo.get(self.img_id)
        else:
            return None



if __name__ == '__main__':
    # 相机位置：3 1 2
    namelis = ['S002C001P003R001A%03d' % i for i in range(1, 61)]
    S = 'e' * 12 + 'f' * 6
    si = int(namelis[0][2:4])
    zp = zipfile.ZipFile('%s:/nturgbd_depth_masked_s%03d.zip' % (S[si], si))

    for name in namelis:
        path = 'nturgb+d_depth_masked/' + name + '/'
        fn = subdir(path, zp.namelist())
        for subname in fn[::None]:
            image = cv2.imdecode(np.array(bytearray(zp.read(subname))), -1)
            image = (image.astype('float32') / 16).clip(0, 255).astype('uint8')
            cv2.putText(image, name[-4:], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2, cv2.LINE_AA)
            cvm.imshow('d', image)
            cv2.waitKey(1)

            image2 = cv2.imdecode(np.fromstring(zp.read(fn[0]), 'uint8'), -1)
            cvm.imshow('im2', (image2.astype('float32') / 16).clip(0, 255).astype('uint8'))
            cv2.waitKey(50)
        cv2.waitKey()

    zp.close()
