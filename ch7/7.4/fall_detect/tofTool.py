# -*- coding: utf-8 -*-
import cv2
import dmcam
import os

def makedirs(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


class TofCapture:
    def __init__(self, rname, record=1, folder=r'../fallData', bgNum=10):

        self.rname = rname
        self.record = record
        self.folder = folder
        self.bgNum = bgNum

        if record:
            if not makedirs(folder + '/%s/dep/' % rname):
                print('exit')
                exit(1)
            makedirs(folder + '/%s/ir/' % rname)
            if bgNum:
                makedirs(folder + '/%s/bg/' % rname)

        self.count = 0
        self.num = 0

    def tof_cap(self, dep0, amp0, show=0):
        img_dep = dep0.clip(0, 65535).astype('uint16')
        img_amp = amp0.clip(0, 65535).astype('uint16')

        rname = self.rname
        self.count += 1
        if self.record:
            if self.count <= self.bgNum:
                cv2.imwrite(self.folder + "/%s/bg/dep_bg_%03d.png" % (rname, self.count), img_dep)
                cv2.imwrite(self.folder + "/%s/bg/ir_bg_%03d.png" % (rname, self.count), img_amp)
            else:
                cv2.imwrite(self.folder + "/%s/dep/dep_%06d.png" % (rname, self.num), img_dep)
                cv2.imwrite(self.folder + "/%s/ir/ir_%06d.png" % (rname, self.num), img_amp)
                self.num += 1

        if show:
            amp_show = cv2.convertScaleAbs(amp0, None, 1)
            dep_show = cv2.convertScaleAbs(dep0, None, 1 / 16)
            dep_show = cv2.merge([dep_show] * 3)

            cutx = 10
            if self.count > self.bgNum:
                cv2.rectangle(dep_show, (cutx, 2), (320 - cutx, 238), color=[0, 255, 0], thickness=2)

            cv2.imshow('Original depth map', dep_show)
            cv2.imshow('Original amp', amp_show)


def read_tof_params(filename):
    params = []
    with open(filename, 'r', encoding='utf-8') as fileid:
        for line in fileid:
            params.append(int(line.split()[0]))  # no of the recorded frames
    return params
