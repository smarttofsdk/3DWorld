#!/usr/bin/python3
# coding=utf-8

import sys, os, math

sys.path.append('./')
from geo_tools              import *
from depth_img_view_cv      import *
from depth_img_proc_cv      import *

from dmcam_dev              import *

####################
# 安装配置工具
# 使用方法
# 1. 安装好SmartTOF，开启电源，连接USB
# 2. 开启本应用程序，等待大约5分钟预热
# 3. 在地面摆放5张A4纸，位置如图所示，放
#    置在4个数字对应位置
#    +-----------------------+
#    | 2                   3 |
#    |                       |
#    |                       |
#    |           1           |
#    |                       |
#    |                       |
#    | 4                   5 |
#    +-----------------------+
#    注意：放置A4纸时，需要尽量和屏幕上显示的白框对齐
#    放置完成后，屏幕上会实时显示这4个位置测量得到的距离值
# 4. 调节相机安装角度和高度，使得
#    1）屏幕上显示的1号纸的距离为1.8M
#    2）2，3，4，5号纸的距离值尽量相同
# 5. 完成步骤4之后，按按键s保存校准结果并退出
#
####################

SKIP_LIST=None
PLAYBACK_END=None

from box_calib import meas_conv_c
import time

def draw_cross_box(x0,y0,x1,y1):
    viewer.draw_line(x0,y0,x1,y0,line_wid=2);
    viewer.draw_line(x0,y1,x1,y1,line_wid=2);
    viewer.draw_line(x0,y0,x0,y1,line_wid=2);
    viewer.draw_line(x1,y0,x1,y1,line_wid=2);
    viewer.draw_line(x0,y0,x1,y1,line_wid=2);
    viewer.draw_line(x1,y0,x0,y1,line_wid=2);

gnd_hgt=0
alpha=0.95
d1f=d2f=d3f=d4f=d5f=0
if __name__ == '__main__':

    print_config()

    # GUI显示器
    viewer = cv_viewer_c(pan_hgt=IMG_HGT, pan_wid=IMG_WID)
    # 校准
    meas_conv = meas_conv_c('./calib_data/')

    # 相机
    if PLAYBACK:
        cam = playback_cam_c(fname_dep=PLAYBACK_DEP, fname_amp=PLAYBACK_AMP, \
                             skip=PLAYBACK_SKIP, rewind=PLAYBACK_REWIND, \
                             skip_list=SKIP_LIST)
    else:
        cam = dmcam_dev_c()
        cam.set_intg(DMCAM_INTG)
        cam.set_framerate(DMCAM_FRAMERATE)
        cam.set_freq(DMCAM_FREQ)
        cam.set_pix_calib(DMCAM_PIX_CALIB)
        cam.set_hdr(DMCAM_HDR)
        #cam.set_gauss_filter(DMCAM_GAUSS)
        cam.set_median_filter(DMCAM_MED)
        if DMCAM_MODE == '2DCS':
            dmfilter = dmcam.filter_args_u()
            dmfilter.sport_mode = 0
            dmcam.filter_enable(cam.dev, dmcam.DMCAM_FILTER_ID_SPORT_MODE, dmfilter, 0)
        else:
            dmcam.filter_disable(cam.dev, dmcam.DMCAM_FILTER_ID_SPORT_MODE)

    # 测量主循环
    frame_cnt = 0
    flag_passed = False
    pass_count = 0
    ave_info = None
    start_time = time.time()
    time.sleep(0.03)
    fps_list = [30, 30, 30]
    while True:
        # 获取图片
        img_dep, img_amp, frame_cnt_new = cam.get_dep_amp()
        if frame_cnt == frame_cnt_new:
            time.sleep(0.001)
            continue
        else:
            frame_cnt = frame_cnt_new
            if frame_cnt % 10 == 0: print('[%d]' % frame_cnt)

        # 有效区域屏蔽码
        mask = np.ones_like(img_dep, dtype=bool) if BACKGROUND_MASK is None else BACKGROUND_MASK.copy()

        # 镜头去畸变
        if MEAS_UNDISTORT:
            img_dep = cv2.undistort(img_dep, MEAS_DMAT, MEAS_DVEC)
            img_amp = cv2.undistort(img_amp, MEAS_DMAT, MEAS_DVEC)
            mask = np.bitwise_and(mask, img_dep > 0)

        if MEAS_DEP_TO_Z: img_dep = img_dep_to_z(img_dep, MEAS_F, MEAS_CX, MEAS_CY)  # 测距射线长度矫正

        # 图像预处理
        mask = np.bitwise_and(mask, img_dep < IMG_DEP_SAT_TH)           # 最大深度过滤
        
        # 显示深度图和强度图
        img_dep_view = img_to_cmap(img_dep.copy(), mask=None, vmin=VIEW_DMIN, vmax=VIEW_DMAX)
        viewer.update_pan_img_rgb(img_dep_view)
        
        # 画测量框
        draw_cross_box(135,105,185,135)
        draw_cross_box(135,  0,185, 30)
        draw_cross_box(290, 95,320,145)
        draw_cross_box(135,210,185,240)
        draw_cross_box(  0, 95, 30,145)
        
        
        # 测量深度
        d1=(img_dep[105:135,135:185]*mask[105:135,135:185]).sum()/mask[105:135,135:185].sum()
        d2=(img_dep[  0: 30,135:185]*mask[  0: 30,135:185]).sum()/mask[  0: 30,135:185].sum()
        d3=(img_dep[ 95:145,290:320]*mask[ 95:145,290:320]).sum()/mask[ 95:145,290:320].sum()
        d4=(img_dep[210:240,135:185]*mask[210:240,135:185]).sum()/mask[210:240,135:185].sum()
        d5=(img_dep[ 95:145,  0: 30]*mask[ 95:145,  0: 30]).sum()/mask[ 95:145,  0: 30].sum()
        
        if math.isnan(d1f) or math.isinf(d1f) or math.isnan(d1) or math.isinf(d1): 
            d1f=0 
        else: 
            d1f=d1f*alpha+(1.0-alpha)*d1
        if math.isnan(d2f) or math.isinf(d2f) or math.isnan(d2) or math.isinf(d2): 
            d2f=0 
        else: 
            d2f=d2f*alpha+(1.0-alpha)*d2
        if math.isnan(d3f) or math.isinf(d3f) or math.isnan(d3) or math.isinf(d3): 
            d3f=0 
        else: 
            d3f=d3f*alpha+(1.0-alpha)*d3
        if math.isnan(d4f) or math.isinf(d4f) or math.isnan(d4) or math.isinf(d4): 
            d4f=0 
        else: 
            d4f=d4f*alpha+(1.0-alpha)*d4
        if math.isnan(d5f) or math.isinf(d5f) or math.isnan(d5) or math.isinf(d5): 
            d5f=0 
        else: 
            d5f=d5f*alpha+(1.0-alpha)*d5


        # 显示测量深度
        viewer.draw_text('%.3f'%d1f,130, 95,font_size=0.6)
        viewer.draw_text('%.3f'%d2f,130, 50,font_size=0.6)
        viewer.draw_text('%.3f'%d3f,260, 85,font_size=0.6)
        viewer.draw_text('%.3f'%d4f,130,200,font_size=0.6)
        viewer.draw_text('%.3f'%d5f, 10, 85,font_size=0.6)

        _,_,gnd_hgt0=meas_conv.conv(0,0,GROUND_DIST-d1)
        if math.isnan(gnd_hgt0) or math.isinf(gnd_hgt0) or \
           math.isnan(gnd_hgt ) or math.isinf(gnd_hgt ): 
            gnd_hgt=0
        else:
            gnd_hgt=gnd_hgt*alpha+(1.0-alpha)*gnd_hgt0
        #print('gnd_hgt:',gnd_hgt)
        viewer.draw_text('%.3f'%gnd_hgt,130,155,color=(0,255,200),font_size=0.6)
        
        # 屏幕刷新
        viewer.update()
        evt, param = viewer.poll_evt()
        if evt is not None: 
            if evt == 'save':
                fname='./config/user_calib_param.txt' 
                fp=open(fname,'wt')
                s='%.12f\n'%gnd_hgt
                fp.write(s)
                fp.close()
                print('[INF] write user calib param to file:',fname)
                print('[INF]    ',s)
                break

        if PLAYBACK_END is not None:
            if frame_cnt == PLAYBACK_END: break
