#coding=utf-8
import logging.handlers
import queue
import struct
import sys
import os
import time
# import win32api
# import win32con

import dmcam
import numpy as np
import cv2
from static_hand import static
from global_cfg      import *
from pyg_viewer      import *
from simple_main import mainloop_trk, draw_trace_mode
from algo            import *
from cfg import *

def update_win(x_list:list,x,win_len=FTIP_TRK_LIST_SZ):
    if len(x_list) < win_len:
        x_list.append(x)
    else:
        del x_list[0]
        x_list.append(x)
    return x_list

cfg_log_level = "info"
cfg_log_filename = "hardware.log"

log_level_table = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARN, "error": logging.ERROR}
log_level = log_level_table[cfg_log_level]

log_fh = logging.handlers.RotatingFileHandler(cfg_log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
log_fh.setLevel(logging.DEBUG)
log_ch = logging.StreamHandler()
log_ch.setLevel(log_level)

log_format_verbose = logging.Formatter(
'%(asctime)s [%(levelname)5s][%(name)8s][%(filename)s:%(lineno)d] %(message)s')
log_fh.setFormatter(log_format_verbose)
log_ch.setFormatter(log_format_verbose)

log = logging.getLogger('Hardware')
log.setLevel(log_level)
log.addHandler(log_fh)
log.addHandler(log_ch)

pos_win = [0,0,0]
label = ['None', 'big5', 'vict', 'fist', 'bravo']

class SmarttofHardware:
    def __init__(self):
        self.dev = None
        self._run = True
        self.status = False  # flag indicate hardware connectivity
        self.empty_frame_cnt = 0  # count the number of empty frames
        self.ENABLE_VIEW = ENABLE_VIEW
        # self.ENABLE_ZMQ = ENABLE_ZMQ
        # self.ENABLE_SOCKET = ENABLE_SOCKET
        # self.ENABLE_JS = ENABLE_JS
        # self.state_seq = []
        if getattr(sys, 'frozen', False):
            res_path = sys._MEIPASS
        else:
            res_path = os.path.split(os.path.realpath(__file__))[0]
        bg_file = "%s/img/BATCbackground.png" % res_path
        self.bg = cv2.imread(bg_file,-1)
        self.bg = cv2.resize(self.bg,(1440,810))
        self.bg = cv2.cvtColor(self.bg, cv2.COLOR_RGB2BGR)

        if self.ENABLE_VIEW:
            self.viewer = pyg_viewer_c(pan_wid=1440, pan_hgt=810, pan=(1, 1))
        else:
            self.viewer = None
        self.algo = ftip_trk_c()
        self.frame_cnt = 0
        self.result_list = []

    def _init_dmcam(self):
        dmcam.init(None)
        dmcam.log_cfg(dmcam.LOG_LEVEL_INFO,
                      dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

        log.info("Scanning dmcam device ..")
        while True:
            devs = dmcam.dev_list()
            if devs is None:
                log.info("No device found")
                print(u"没有找到TCM-E2设备，请插入设备后点任意键继续")
                input("")
                # win32api.MessageBox(0, u"没有找到TCM-E2设备，请插入设备后点击【确定】继续", u"警告", win32con.MB_OK)
                # time.sleep(1)
            else:
                log.info("Found {} devices".format(len(devs)))
                break


    def _start_dmcam(self):
        if self.dev is None:
            log.info(" Open dmcam device ..")
            self.dev = dmcam.dev_open(None)
            if self.dev.version.ver.sw_ver < 154:
                print(u"仅支持固件版本不低于于154的模组设备，当前固件版本{},请升级模组固件，详见Readme".format(self.dev.version.ver.sw_ver))
                input("")
                # win32api.MessageBox(0, u"仅支持固件版本不低于于154的模组设备，当前固件版本{},请升级模组固件，详见Readme".format(self.dev.version.ver.sw_ver), u"错误", win32con.MB_OK)
                sys.exit(0)
            # - set capture config  -
            cap_cfg = dmcam.cap_cfg_t()
            cap_cfg.cache_frames_cnt = 10  # framebuffer = 10
            cap_cfg.on_error = None  # use cap_set_callback_on_error to set cb
            cap_cfg.on_frame_rdy = None  # use cap_set_callback_on_frame_ready to set cb
            cap_cfg.en_save_replay = False  # True = save replay, False = not save
            cap_cfg.en_save_dist_u16 = False  # True to save dist stream for openni replay
            cap_cfg.en_save_gray_u16 = False  # True to save gray stream for openni replay
            cap_cfg.fname_replay = os.fsencode("dm_replay.oni")  # set replay filename

            log.info("Set parameters ...")
            wparams = {
                dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
                dmcam.PARAM_FRAME_RATE: dmcam.param_val_u(),
                dmcam.PARAM_FRAME_FORMAT: dmcam.param_val_u(),
                # dmcam.PARAM_HDR_INTG_TIME: dmcam.param_val_u(),
                dmcam.PARAM_MOD_FREQ: dmcam.param_val_u()
            }
            
            wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = 250
            # wparams[dmcam.PARAM_HDR_INTG_TIME].intg.intg_us = 1000
            wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = 20
            wparams[dmcam.PARAM_FRAME_FORMAT].frame_format.format = 2
            wparams[dmcam.PARAM_MOD_FREQ].mod_freq = 12000000
            amp_min_val = dmcam.filter_args_u()
            amp_min_val.min_amp = 60
            intg_auto_arg = dmcam.filter_args_u()
            intg_auto_arg.sat_ration = 100  # 自动曝光设置的值

            if not dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val,
                                       sys.getsizeof(amp_min_val)):
                log.error("set amp to %d %% failed" % 0)
            if not dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_MEDIAN):
                log.error("disable Median Filter failed")
            # hdr = dmcam.filter_args_u()
            if not dmcam.filter_disable(self.dev, dmcam.DMCAM_FILTER_ID_HDR):
                log.error("set hdr failed")
            if not dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_AUTO_INTG, intg_auto_arg, sys.getsizeof(intg_auto_arg)):
                log.error("set auto intg failed")
                print("set auto intg failed")
            else:
                print("set auto intg")

            if not dmcam.param_batch_set(self.dev, wparams):
                log.error("set parameter failed")
            assert self.dev is not None
        if not self.status:
            log.info("Start capture ...")
            dmcam.cap_start(self.dev)
            self.status = True

    def _stop_dmcam(self):
        if self.status:
            log.info("Stop capture ...")
            dmcam.cap_stop(self.dev)
            self.status = False

    def _close_dmcam(self):
        if self.status:
            log.info("Close device ...")
            self.status = False
            dmcam.dev_close(self.dev)
            self.dev = None
        dmcam.uninit()

    def get_frames(self):
        if self.dev is None:
            log.warning("Device not initialized, retry to capture in 0.5s!")
            time.sleep(0.5)
            return
        f = bytearray(320 * 240 * 4 * 2)
        finfo = dmcam.frame_t()
        log.debug("dmcam.cap_get_frames")
        ret = dmcam.cap_get_frames(self.dev, 1, f, finfo)
        w = finfo.frame_info.width
        h = finfo.frame_info.height

        intg_auto_arg = dmcam.filter_args_u()
        intg_auto_arg.sat_ratio = 5
        if not dmcam.filter_enable(self.dev, dmcam.DMCAM_FILTER_ID_AUTO_INTG, intg_auto_arg,
                                   sys.getsizeof(intg_auto_arg)):
            print(" enable AUTO INTG filter (sat_ratio=%d) failed" % intg_auto_arg.sat_ratio)

        if ret <= 0:
            if ret == -10:
                log.warning("dmcam USB cable unplugged, try to reconnect!")
                self._close_dmcam()
                self._init_dmcam()
                self._start_dmcam()
            elif ret == 0:
                self.empty_frame_cnt += 1
                if self.empty_frame_cnt > 20:
                    log.warning("dmcam capture timeout, restart capture")
                    self._stop_dmcam()
                    self._start_dmcam()
                    self.empty_frame_cnt = 0
            else:
                log.warning("dmcam capture error, errono:{}".format(ret))
                self._stop_dmcam()
                self._start_dmcam()

        else:
            self.empty_frame_cnt = 0
            print()
            dist_cnt, dist = dmcam.frame_get_distance(self.dev, w * h, f, finfo.frame_info)
            # gray_cnt, gray = dmcam.frame_get_gray(self.dev, w * h, f, finfo.frame_info)
            # gray_data = gray.astype(np.uint16)
            # gray = ((gray/16).astype(np.uint8)).reshape(240,320)
            
            if dist_cnt == w*h:
                data = (dist*1000).astype(np.uint16)
                frame = data.reshape(240, 320)

                cv2.imshow('frame', cv2.convertScaleAbs(frame, None, 1/20))
                cv2.waitKey(1)

                # ## 静态手势
                img_s = frame.copy()
                result_s, result_str = static(img_s)
                self.result_list = update_win(self.result_list, result_s, win_len=5)

                ## 动态手势
                img_d = frame.copy()
                img_d = img_d.astype(np.float32) / 10000
                self.algo = mainloop_trk(self.algo, img_d, self.viewer,self.bg)

                self.viewer.put_text(str('手势识别结果为：'),150,1000)

                if result_s > 0 and self.algo.decision == 'point':
                    if self.ENABLE_VIEW:
                        # if len(self.result_list)>0:
                        #     counts = np.bincount(self.result_list)
                        #     idx = np.argmax(counts)
                        #     if idx > 0:
                        #         self.viewer.put_text(str(label[idx]),200,1050)
                        self.viewer.put_text(str(result_str),200,1000)
                    # final_result = result_str
                else:
                    x, y = self.algo.xlist[self.algo.idx], self.algo.ylist[self.algo.idx]
                    if self.ENABLE_VIEW:
                        if self.algo.trace_pattern['type'] != 'point':
                            self.viewer.draw_circle(int(x*2+400), int(y*2+165), r=5, line_wid=5)
                            draw_trace_mode(self.viewer, mode=self.algo.trace_pattern['type'], pos=self.algo.trace_pattern['param'])
                            if self.algo.trace_pattern['type'] == 'circle':
                                self.viewer.put_text(str('转圈'), 215, 1050)
                            elif self.algo.trace_pattern['type'] == 'h-line':
                                self.viewer.put_text(str('水平移动'), 215, 1050)
                            elif self.algo.trace_pattern['type'] == 'v-line':
                                self.viewer.put_text(str('竖直移动'), 215, 1050)
                            # self.result_list=[]
                    # if self.algo.trace_pattern['type'] == 'point':
                    #     final_result = 'None'
                    # else:
                    #     final_result = self.algo.trace_pattern['type']
                    #     if final_result == "circle":
                    #         pos = self.algo.trace_pattern['param']
                    #         update_win(pos_win, pos, win_len=3)
                    #         if pos_win[0] < pos_win[1] and pos_win[1]<pos_win[2]:
                    #             final_result = "c-circle"
                    #         elif pos_win[0] > pos_win[1] and pos_win[1] > pos_win[2]:
                    #             final_result = "circle"
                    #     if final_result == "h-line":
                    #         pos = self.algo.trace_pattern['param']
                    #         update_win(pos_win, pos, win_len=3)
                    #         if pos_win[0] < pos_win[1] and pos_win[1]<pos_win[2]:
                    #             final_result = "right"
                    #         elif pos_win[0] > pos_win[1] and pos_win[1] > pos_win[2]:
                    #             final_result = "left"
                    #     if final_result == "v-line":
                    #         pos = self.algo.trace_pattern['param']
                    #         update_win(pos_win, pos, win_len=3)
                    #         if pos_win[0] < pos_win[1] and pos_win[1]<pos_win[2]:
                    #             final_result = "down"
                    #         elif pos_win[0] > pos_win[1] and pos_win[1] > pos_win[2]:
                    #             final_result = "up"

                # 刷新屏幕显示
                if self.ENABLE_VIEW:
                    self.viewer.update()
                    evt, param = self.viewer.poll_evt()
                    if evt == 'quit':
                        return False
        return True

    def run(self):
        self._init_dmcam()
        self._start_dmcam()
        log.info("cam daemon started...")
        while self._run:
            if not self.get_frames():
                break

        self.viewer.close()


if __name__ == "__main__":
    sh = SmarttofHardware()
    sh.run()


