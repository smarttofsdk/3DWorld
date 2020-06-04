# -*- coding: utf-8 -*-
import sys
from filetool import *
# from tofTool import *
import os
import numpy as np
import cv2
import time

IMG_SHAPE = (240, 320)
# IMG_SHAPE=(320,240)   #横拍用
# IMG_SHAPE=(424,512)

IMG_HGT, IMG_WID = IMG_SHAPE
IMG_SZ = IMG_WID * IMG_HGT

# 采样点生成
maxpf = 150
spf = 2
spMat = np.ones((spf * 2 + 1, spf * 2 + 1))
ux, uy = np.where(spMat > 0)
ux = (ux - spf) * (maxpf / spf) * 1000
uy = (uy - spf) * (maxpf / spf) * 1000

# 关节点连接顺序--TOF自标注骨架
connect_diedao = [-1, 2, 1, 2, 2, 4, 5, 2, 7, 8, 3, 10, 11, 3, 13, 14]
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

# 关节点连接顺序--NTU骨架
connecting_joint = [-1, 2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
# 1  2  3   4  5   6  7  8  9  10  11  12  13 14  15  16  17 18  19  20  21 22 23 24  25
# 简化的关节点
mask_joint = [1, 2, 21, 4, 13, 17, 5, 9, 14, 15, 18, 19, 6, 7, 10, 11]


## 功能描述：
#     点云变换
# 输入参数：
#     pc: 输入点云集合，每个点对应一行数据坐标(x,y,z)
#     T     ：变换矩阵（4x4）
# 输出参数：
#     pc_out：输出点云集合，每个点对应一行数据坐标(x,y,z)
def pc_trans(T, pc):
    T_rot = T[0:3, 0:3]  # 截取旋转部分
    pc_out = np.dot(pc, T_rot)  # 计算旋转
    pc_out[:, 0] += T[3, 0]  # 计算平移
    pc_out[:, 1] += T[3, 1]
    pc_out[:, 2] += T[3, 2]

    return pc_out


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
def pc_rotx(b, pc=[]):
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(b), np.sin(b), 0],
                  [0, -np.sin(b), np.cos(b), 0],
                  [0, 0, 0, 1]])  # 绕X轴旋转
    return pc_trans(T, pc) if len(pc) else T


def transxy(p, imgd):
    """
    # 像素坐标转换为xyz坐标

    :param p: 像素坐标
    :param imgd: 深度图
    :return: xyz坐标
    """
    fxy = 260.64507
    u0, v0 = 160, 120
    u, v = p
    d = imgd[v, u] / 1000.0

    x = (u - u0) * d / fxy
    y = (v - v0) * d / fxy
    z = d
    pc_out = np.array([[x, y, z]])
    return pc_out


class CalMeanSpeed:
    def __init__(self, frameDelay=4, rx=-30):
        """
        :param frameDelay: 计算平均速度的帧数
        :param rx: 角度，相机位姿
        """
        self.frameDelay = frameDelay
        self.T = pc_rotx(rx / 180 * 3.14159)
        self.storeV = np.zeros((frameDelay, 3))
        self.fi = 0
        self.lastpc = np.zeros((1, 3))
        self.dt = TimeRecord()

    def meanSpeed(self, joint, imgd):
        """
        # 计算平均速度
        # 图像坐标转换为真实空间xyz坐标
        # 与上一帧作差/时间,计算速度并存储
        # 计算平均速度

        :param joint: 关节坐标
        :param imgd: 深度图像
        :return: 平均速度
        """
        pc = transxy(joint, imgd)
        dv = (pc - self.lastpc) / self.dt.timePass()
        dv = pc_trans(self.T, dv)
        self.storeV[self.fi] = dv[0]

        self.fi = (self.fi + 1) % self.frameDelay
        self.lastpc = pc
        return np.mean(self.storeV, 0)


class TimeRecord:
    def __init__(self):
        self.start_time = self.last_time = time.time()

    def timePass(self):
        ct = time.time()
        dt = ct - self.last_time
        self.last_time = ct
        return dt

    def totelTime(self):
        ct = time.time()
        totaltime = ct - self.start_time
        # print('Total Time:', (totaltime), 's')
        return totaltime

    def getmtime(self, filename):
        """Return the last modification time of a file, reported by os.stat()."""
        return os.stat(filename).st_mtime


## 获取关节点坐标--适配tof图片分辨率
def changeJoints_ki2tof(body):
    joints = {}
    for i in body:
        dx = (body[i][0] - 255.5) * 3.2969729e2 / 3.672420124e2 + 159.5
        dy = (body[i][1] - 211.5) * 3.2969729e2 / 3.672420124e2 + 119.5
        joints[i] = [dx, dy]
    return joints


def connect_joints(img_clf, joints, s=0, mode='ntu'):
    """
    # 连接关节点，可视化

    :param img_clf: 输入图像
    :param joints: 关节点坐标
    :param s: 输入图像类型
    :param mode:
        # mode=='ntu': 符合ntu骨架
        # 'diedao': 自标注骨架
    :return: 关节点连接完后的图像
    """

    if s == 1:
        frame = cv2.merge([img_clf] * 3)
    elif s == 2:
        frame = cv2.applyColorMap(img_clf, cv2.COLORMAP_HOT)
    else:  # s == 0:
        frame = img_clf.copy()

    for j in joints:
        joint = joints[j]
        dx, dy = joint

        if mode == 'diedao':
            k = connect_diedao[j]
            joint2 = joints.get(k)
        else:  # mode == 'ntu':
            k = connecting_joint[j]
            joint2 = joints.get(k) if k != 3 else joints.get(k, joints.get(21))

        if joint2 is not None:
            dx2, dy2 = joint2
            if dx == dx2: dx2 += 0.001
            if dy == dy2: dy2 += 0.001

            xdist = abs(dx - dx2)
            ydist = abs(dy - dy2)

            # locate the pixels of the connecting line between the two joints
            if xdist > ydist:
                xrange = np.arange(dx, dx2, np.sign(dx2 - dx))
                yrange = np.arange(dy, dy2, np.sign(dy2 - dy) * ydist / xdist)
            else:
                yrange = np.arange(dy, dy2, np.sign(dy2 - dy))
                xrange = np.arange(dx, dx2, np.sign(dx2 - dx) * xdist / ydist)
            # draw the line!
            # use red color for drawing joint connections 画连接线
            BGR = (0, 0, 255)
            cx = 2
            xrange = xrange.round().astype('int')
            yrange = np.int32(np.round(yrange))
            for i in range(min(len(xrange), len(yrange))):
                dx, dy = xrange[i], yrange[i]
                for t in range(3):
                    frame[dy - cx:dy + cx, dx - cx:dx + cx, t] = BGR[t]

        # use green color to draw joints 画关节点
        joint = joints[j]
        dx = int(round(joint[0]))
        dy = int(round(joint[1]))

        BGR = (0, 255, 0)
        cx = 4
        for t in range(3):
            frame[dy - cx:dy + cx, dx - cx:dx + cx, t] = BGR[t]

    return frame


def label_joints_area(img_dep, bodys, wid=5, mode='0'):
    """
    ##功能描述：
    # 为关节点附近的点打标签得到训练标签图像

    :param img_dep: 深度图像
    :param bodys: 对应的关节点坐标数据
    :param wid: 采样点范围
    :param mode: 'ki2tof', '0'
    :return: 训练标签图像
    """

    # wid = 15
    cl = np.zeros_like(img_dep, dtype=np.uint8)

    # for 第1个 the detected skeletons in the current frame:
    for b in bodys[0:1]:
        # for all the joints need to be labeled:
        for j in b:
            if j in mask_joint:
                dx, dy = b[j]
                if mode == 'ki2tof':
                    dx = round((dx - 255.5) * 3.2969729e2 / 3.672420124e2 + 159.5)
                    dy = round((dy - 211.5) * 3.2969729e2 / 3.672420124e2 + 119.5)
                else:  # mode=='0':
                    dx = round(dx)
                    dy = round(dy)

                mask = (img_dep[max(dy - wid, 0): dy + wid, max(dx - wid, 0): dx + wid] > 80) * j
                cl[max(dy - wid, 0): dy + wid, max(dx - wid, 0): dx + wid] = mask

    return np.uint8(cl)


def hand_cut(img_dep=[], img_amp=[], dmin=-1, dmax=-1, amp_th=-1, cutx=0, cuty=0, mask=[]):
    """
    ## 根据距离门限和亮度阈值切割
    :param img_dep: 深度图
    :param img_amp: 强度图
    :param dmin,dmax,amp_th 若<0则表示无效
    :return: mask_out
    """
    mask_out = np.ones_like(img_dep, 'uint8') if len(mask) == 0 else mask.copy()

    # 切除过暗的像素
    if amp_th >= 0: mask_out[img_amp < amp_th] = 0

    # 切除距离区间外的像素
    if dmin >= 0: mask_out[img_dep < dmin] = 0
    if dmax >= 0: mask_out[img_dep > dmax] = 0

    # 切除图像四周区域
    if cutx > 0:
        mask_out[:, :cutx] = 0
        mask_out[:, -cutx:] = 0
    if cuty > 0:
        mask_out[:cuty, :] = 0
        mask_out[-cuty:, :] = 0

    return mask_out


def find_longest_contour(c):
    max = 0
    index = 0
    for i in range(len(c)):
        if max < c[i].size:
            max = c[i].size
            index = i
    return c[index]


def calc_contour(img_bw, gen_mask=False):
    _, contours, _ = cv2.findContours(img_bw * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        if gen_mask:
            return [], [], [], []
        else:
            return [], []

    contours = find_longest_contour(contours)
    area_mask = np.zeros((IMG_HGT, IMG_WID)).astype(np.uint8)
    cv2.drawContours(area_mask, [contours], -1, 255, thickness=cv2.FILLED)

    contours = np.array(contours).flatten()  # 轮廓坐标序列（x/y间隔存放）
    x_contour, y_contour = contours[0::2], contours[1::2]
    x_contour, y_contour = x_contour[::2], y_contour[::2]

    # 轮廓曲率强度图（2D)
    if gen_mask:
        mask = np.full((IMG_HGT, IMG_WID), False, dtype=bool)
        mask[y_contour, x_contour] = True
        return x_contour, y_contour, mask, area_mask
    else:
        return x_contour, y_contour

