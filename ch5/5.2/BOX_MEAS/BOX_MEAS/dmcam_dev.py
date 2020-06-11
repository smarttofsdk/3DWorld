#!/usr/bin/python
# coding=utf-8

import sys, os, time

import numpy as np
import cv2
from cv_viewer import *
import dmcam


class dmcam_dev_c:
    def __init__(self):
        dmcam.init(None)
        dmcam.log_cfg(dmcam.LOG_LEVEL_INFO, dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

        self.dev = dmcam.dev_open(None)
        assert self.dev is not None
        
        # - set capture config  -
        cap_cfg = dmcam.cap_cfg_t()
        cap_cfg.cache_frames_cnt = 10  # framebuffer = 10
        cap_cfg.on_error = None        # use cap_set_callback_on_error to set cb
        cap_cfg.on_frame_rdy = None    # use cap_set_callback_on_frame_ready to set cb
        cap_cfg.en_save_replay = True  # True = save replay, False = not save
        cap_cfg.en_save_dist_u16 = False # True to save dist stream for openni replay
        cap_cfg.en_save_gray_u16 = False # True to save gray stream for openni replay
        cap_cfg.fname_replay = os.fsencode("dm_replay.oni")  # set replay filename
        
        dmcam.cap_config_set(self.dev, cap_cfg)
        #dmcam.cap_set_frame_buffer(self.dev, None, 10 * 320 * 240 * 4 * 2)

        self.frame_data = bytearray(320 * 240 * 4 * 4)
        self.frame_dist = [np.zeros((240, 320))]
        self.frame_gray = [np.zeros((240, 320))]
        self.frame_cnt = 0

        self.start()
        return

    def start(self):
        print('[INF] dmcam_dev_c.start()')
        dmcam.cap_start(self.dev)

    def stop(self):
        print('[INF] dmcam_dev_c.stop()')
        dmcam.cap_stop(self.dev)

    def close(self):
        print('[INF] dmcam_dev_c.close()')
        dmcam.dev_close(self.dev)
        dmcam.uninit()
        return

    def poll_frame(self):
        finfo = dmcam.frame_t()
        ret = dmcam.cap_get_frame(self.dev, self.frame_data, finfo)
        if ret > 0:
            w = finfo.frame_info.width
            h = finfo.frame_info.height

            _, frame_dist = dmcam.frame_get_distance(self.dev, w * h, self.frame_data, finfo.frame_info)
            _, frame_gray = dmcam.frame_get_gray(self.dev, w * h, self.frame_data, finfo.frame_info)

            self.frame_dist.append(frame_dist.reshape(h, w))
            self.frame_gray.append(frame_gray.reshape(h, w))

            if len(self.frame_dist) > 4: self.frame_dist.pop(0)
            if len(self.frame_gray) > 4: self.frame_gray.pop(0)

            self.frame_cnt += 1
        return ret > 0

    def get_dist(self):
        return self.frame_dist[-1]

    def get_gray(self):
        return self.frame_gray[-1]

    def get_dist_gray(self):
        return self.frame_dist[-1], self.frame_gray[-1]

    def get_dep_amp(self):
        self.poll_frame()
        return self.frame_dist[-1], self.frame_gray[-1], self.frame_cnt

    def set_illum(self, v=100):
        print('[INF] dmcam_dev_c.set_illum(%d)' % v)
        wparams = {dmcam.PARAM_ILLUM_POWER: dmcam.param_val_u()}
        wparams[dmcam.PARAM_ILLUM_POWER].illum_power.percent = v
        if not dmcam.param_batch_set(self.dev, wparams):
            print('Set parameter failed')

    def set_intg(self, v=500):
        print('[INF] dmcam_dev_c.set_intg(%d)' % v)
        wparams = {dmcam.PARAM_INTG_TIME: dmcam.param_val_u()}
        wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = v
        if not dmcam.param_batch_set(self.dev, wparams):
            print('Set parameter failed')

    def set_hdr_intg(self, v=500):
        print('[INF] dmcam_dev_c.set_hdr_intg(%d)' % v)
        wparams = {dmcam.PARAM_HDR_INTG_TIME: dmcam.param_val_u()}
        wparams[dmcam.PARAM_HDR_INTG_TIME].intg.intg_us = v
        if not dmcam.param_batch_set(self.dev, wparams):
            print('Set parameter failed')

    def set_hdr(self, v=1):
        print('[INF] dmcam_dev_c.set_hdr(%d)' % v)
        hdr = dmcam.filter_args_u()
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_HDR, hdr, sys.getsizeof(hdr))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_HDR)

    def set_framerate(self, v=30):
        print('[INF] dmcam_dev_c.set_framerate(%d)' % v)
        wparams = {dmcam.PARAM_FRAME_RATE: dmcam.param_val_u()}
        wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = v
        if not dmcam.param_batch_set(self.dev, wparams):
            print('Set parameter failed')

    def set_freq(self, v=12000000):
        print('[INF] dmcam_dev_c.set_freq(%d)' % v)
        wparams = {dmcam.PARAM_MOD_FREQ: dmcam.param_val_u()}
        wparams[dmcam.PARAM_MOD_FREQ].mod_freq = int(v)
        if not dmcam.param_batch_set(self.dev, wparams):
            print('Set parameter failed')

    def set_amp_filter(self, v=300):
        print('[INF] dmcam_dev_c.set_amp_filter(%d)' % v)
        witem = dmcam.filter_args_u()
        witem.min_amp = v
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_AMP, witem, sys.getsizeof(witem))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_AMP)

    def set_gauss_filter(self, v=1):
        print('[INF] dmcam_dev_c.set_gauss_filter(%d)' % v)
        witem = dmcam.filter_args_u()
        witem.min_amp = v
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_GAUSS, witem, sys.getsizeof(witem))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_GAUSS)

    def set_median_filter(self, v=1):
        print('[INF] dmcam_dev_c.set_median_filter(%d)' % v)
        witem = dmcam.filter_args_u()
        witem.median_ksize = v
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_MEDIAN, witem, sys.getsizeof(witem))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_MEDIAN)

    def set_len_calib(self, v=1):
        print('[INF] dmcam_dev_c.set_len_calib(%d)' % v)
        witem = dmcam.filter_args_u()
        witem.lens_id = v
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_LEN_CALIB, witem, sys.getsizeof(witem))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_LEN_CALIB)

    def set_pix_calib(self, v=1):
        print('[INF] dmcam_dev_c.set_pix_calib(%d)' % v)
        witem = dmcam.filter_args_u()
        witem.case_idx = v
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_PIXEL_CALIB, witem, sys.getsizeof(witem))
        else:
            dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_PIXEL_CALIB)
    
    def set_auto_arg(self, v=1):
        print('[INF] dmcam_dev_c.set_auto_arg(%d)' % v)
        intg_auto_arg = dmcam.filter_args_u()
        intg_auto_arg.sat_ratio = v    #自动曝光设置的值
        if v > 0:
            dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_AUTO_INTG, intg_auto_arg, sys.getsizeof(intg_auto_arg))  #开启自动曝光
        else:
            dmcam.filter_disable(self.dev,dmcam.DMCAM_FILTER_ID_AUTO_INTG)             #关闭自动曝光

############
## 单元测试
############
#from pyg_viewer import *
from depth_img_proc_cv import *
from depth_img_view_cv import record_dep_amp, playback_dep_amp


def test_dmcam(mode=-1):
    viewer = cv_viewer_c(pan_wid=320, pan_hgt=240, pan_num=(1, 2))

    cam = dmcam_dev_c()
    cam.set_intg(1500)
    cam.set_framerate(10)
    cam.set_freq(12e6)
    cam.set_pix_calib(1)
    cam.set_hdr(0)
    cam.set_gauss_filter(0)
    cam.set_median_filter(0)

    print('[INF] test_dmcam(%d)' % mode)
    frame_cnt = cam.frame_cnt
    while True:
        if cam.poll_frame():
            frame_cnt = cam.frame_cnt
            if frame_cnt % 25 == 0:
                print('[%d]' % frame_cnt)
                if frame_cnt % 100 == 0:
                    if mode == 0: cam.set_intg(500)
                    if mode == 1: cam.set_framerate(10)
                    if mode == 2: cam.set_freq(12e6)
                    if mode == 3: cam.set_amp_filter(200)
                    if mode == 4: cam.set_len_calib(1)
                    if mode == 5: cam.set_pix_calib(1)
                    if mode == 6:
                        cam.set_hdr(1)
                        cam.set_hdr_intg(100)
                        cam.set_intg(1500)
                    if mode == 7: cam.set_gauss_filter(5)
                    if mode == 8: cam.set_median_filter(3)
                elif frame_cnt % 50 == 0:
                    if mode == 0: cam.set_intg(1200)
                    if mode == 1: cam.set_framerate(20)
                    if mode == 2: cam.set_freq(24e6)
                    if mode == 3: cam.set_amp_filter(300)
                    if mode == 4: cam.set_len_calib(0)
                    if mode == 5: cam.set_pix_calib(1)
                    if mode == 6: pass#cam.set_hdr(1)
                    if mode == 7: cam.set_gauss_filter(0)
                    if mode == 8: cam.set_median_filter(0)
                    pass
        else:
            time.sleep(0.001)
            continue

        img_dep, img_amp = cam.get_dist_gray()
        viewer.update_pan_img_rgb(img_to_cmap(img_dep, mask=None, vmin=0, vmax=5).reshape(IMG_HGT,IMG_WID,3), pan_id=(0, 0))
        viewer.update_pan_img_gray(img_amp * 0.4, pan_id=(0, 1))

        viewer.update()
        evt, param = viewer.poll_evt()
        if evt is not None:
            if evt == 'quit':
                break

    cam.stop()
    cam.close()

import time
import os
if __name__ == '__main__':
    #IMG_WID, IMG_HGT = 320, 240
    if False:
        test_dmcam(mode=6)
    time_stamp=time.strftime('%d_%H_%M',time.localtime(time.time()))
    if True:
        cam = dmcam_dev_c()
        cam.set_intg(500)
        cam.set_framerate(30)
        cam.set_freq(24e6)
        cam.set_pix_calib(1)
        '''cam.set_hdr(0)
        cam.set_hdr_intg(1500)
        cam.set_intg(100)
        cam.set_auto_arg(5)'''
        #cam.set_amp_filter(60)
        #cam.set_median_filter(30)
        if os.path.exists('./data')==False:
            os.mkdir('./data')
        #开始采集
        record_dep_amp(cam, './data/dep_'+time_stamp+'.bin', './data/amp_'+time_stamp+'.bin', num=10, auto_save=True)
        #关闭hdr
        #cam.set_hdr(0)
        #record_dep_amp(cam, './data/dep_'+time_stamp+'.bin', './data/amp_'+time_stamp+'.bin', num=10, auto_save=True)
        '''cam.set_hdr_intg(1500)
        cam.set_intg(100)'''
        #打开自动曝光
        #cam.set_auto_arg(5)
        record_dep_amp(cam, './data/dep_'+time_stamp+'.bin', './data/amp_'+time_stamp+'.bin', num=600)
        exit()

    if False:
        playback_dep_amp('./data/dep_25_16_30.bin', './data/amp_25_16_30.bin', fps=60)
