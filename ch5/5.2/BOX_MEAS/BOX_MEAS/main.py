#!/usr/bin/python3
# coding=utf-8

## 立方体测量

import itertools    as itt
import sys
import os
from   skimage.feature import canny
from   skimage.transform import (hough_line, hough_line_peaks)

sys.path.append('./')
#sys.path.append('./config')
from geo_tools         import *
from depth_img_view_cv import *
from depth_img_proc_cv import *
from dmcam_dev          import *
from rect_func   import *
import scipy.misc
## TODO
# 1. 背景检测跳过前面30帧
# 3. 地面参数计算
# 4. 小物体测量
# 5. 时间序列预测矩形
# 6. 镜头参数计算距离
# 8. 检查消失的矩形
# 10. 用区域外切矩形找到小矩形
#     https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
# 11. 切割出立方体顶面后，精细处理
# 12. 幅度过滤
# 13. 矩形边界匹配根据边长设置比门限（失配长度，不是失配距离）
# 14. 矩形直角歪斜门限根据边长比调整

def load_config(fname='config.txt'):
    fp=open(fname,'rb')
    for cmd in fp:
        print(cmd,end='')
        exec(cmd,globals())

## 矩形匹配


####################
# 入口代码
####################
#            
SKIP_LIST=None #{ 180:880-180 }
PLAYBACK_END=None#480#1200 #
from box_calib import meas_conv_c
import time
from eveluate import do_cprofile
load_config('./config/config.txt')

def main():
    print_config()
    fp_log = open(LOG_FNAME, 'w')
    # GUI显示器
    viewer = cv_viewer_c(pan_hgt=IMG_HGT, pan_wid=IMG_WID, pan_num=(1, 2))

    # 校准
    if MANUAL_CALI == True:
        if os.path.exists('./calib_data')==False:
            main_log()
        else:
            meas_conv = meas_conv_c('./calib_data/')  # './calib_data/'
        
#    if GRAND_DEPTH_CALIBRATE_FROM_FILE  == True:
#        fp_ground=open('./config/ground_calibrate.txt','rt')
#        Ground_Depth=np.fromfile(fp_ground,dtype=PLAY_TYPE,count=IMG_HGT)
    # 加载用户简易校准数据
    if ENABLE_USER_CALIB_PARAM:
        fname='./config/user_calib_param.txt'
        fp=open(fname,'rt')
        for ln in fp:
            ln=ln.strip()
            if len(ln)==0: continue
            print('[INF]    ',ln)
            user_calib_param=eval(ln)
        print('[INF]    load user_calib_param:',user_calib_param)

    # 追踪运动速度
    speed_tk = speed_tracker()

    # 相机
    if PLAYBACK:
        cam = playback_cam_c(fname_dep=PLAYBACK_DEP, fname_amp=PLAYBACK_AMP, img_wid=IMG_WID,img_hgt=IMG_HGT,\
                             skip=PLAYBACK_SKIP, rewind=PLAYBACK_REWIND, skip_list=SKIP_LIST)  # {150:50, 300:50})
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
        else:  # 4dcs mode
            dmcam.filter_disable(cam.dev, dmcam.DMCAM_FILTER_ID_SPORT_MODE)
    # 构建运算和滤波模块
    if FLY_NOISE_FILTER: fly_noise_mask = fly_noise_mask_c()  # 飞散噪声过滤器
    if IIR_FILTER: iir_filter = img_iir_filter_c(alpha=IIR_ALPHA, img_wid=IMG_WID, img_hgt=IMG_HGT)  # 时域平滑滤波器
    if MID_FILTER: mid_filter = mid3_time_filter_c(img_wid=IMG_WID, img_hgt=IMG_HGT)  # 时域中值滤波器
    if RECT_TRK: rect_trk = rect_tracker_c()  # 矩形跟踪器
    if MEAS_FILTER: meas_filter = meas_filter_c(20)
    Img_wid=IMG_WID
    Img_hgt=IMG_HGT
    # 背景检测
    bg_dep, bg_amp, ground_mean, ground_dist, trans_mat = background_det(cam, viewer)

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
        
        if frame_cnt<150: continue

        if PRINT_FPS:
            run_time = time.time() - start_time
            start_time = time.time()
            fps_list.append(1.0 / (run_time + 1e-10))
            fps_list.pop(0)
            fps = np.mean(fps_list)
            print('fps=', fps)

        # 有效区域屏蔽码
        mask = np.ones_like(img_dep, dtype=bool) if BACKGROUND_MASK is None else BACKGROUND_MASK.copy()

        # 镜头去畸变
        if MEAS_UNDISTORT:
            img_dep = cv2.undistort(img_dep, MEAS_DMAT, MEAS_DVEC)
            img_amp = cv2.undistort(img_amp, MEAS_DMAT, MEAS_DVEC)
            mask = np.bitwise_and(mask, img_dep > 0)

        if MEAS_DEP_TO_Z: img_dep = img_dep_to_z(img_dep, MEAS_F, MEAS_CX, MEAS_CY)  # 测距射线长度矫正

        #更新背景深度图
        if frame_cnt%10==0:
            bg_update_mask=np.abs(img_dep-bg_dep)<0.03
            bg_dep[bg_update_mask]=(0.1*img_dep+0.9*bg_dep)[bg_update_mask]
            bg_update_mask = np.abs(img_amp - bg_amp) < 20
            bg_amp[bg_update_mask] = (0.02 * img_amp + 0.98 * bg_amp)[bg_update_mask]
        # 显示深度图和强度图
        img_dep_view = img_to_cmap(img_dep.copy(), mask=None, vmin=VIEW_DMIN, vmax=VIEW_DMAX)
        cv2.putText(img_dep_view, '[%d]' % frame_cnt, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        #viewer.update_pan_img_rgb(img_dep_view, pan_id=(0, 0))
        viewer.update_pan_img_gray(img_amp , pan_id=(0, 0))
        # 图像预处理
        mask = np.bitwise_and(mask, img_dep < IMG_DEP_SAT_TH)  # 最大深度过滤
        if AMP_FILTER: mask = np.bitwise_and(mask, img_amp >= AMP_TH)  # 最小光强过滤
        mask_bg = (bg_dep - img_dep) > BG_NOISE_TH  # 背景扣除， 只保留背景之上的像素

        #if AMP_FILTER: mask_bg = np.bitwise_or(bg_amp < AMP_TH, mask_bg)  # 背景光强过滤，只要够暗都认为是前景
        #mask = np.bitwise_and(mask, mask_bg)
        #cv2.imshow('mask', np.float32(mask))
        # 滤波处理
        if FILL_HOLE: _, mask = img_fill_hole(img_dep, mask, win=3, th=4)  # 填补空洞

        # 边沿检测
        #gray_bg_mask=np.bitwise_and(bg_amp<255,np.bitwise_and(img_amp >100,img_amp<1000))#背景暗且前景不反光，用灰度提取前景

        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
        if EDGE_DEP_BY_AMP:  # 基于强度图的边沿检测,深度图辅助测量
            img_bin_diff= np.abs(img_amp - bg_amp)
            #cv2.imshow('img_bin_diff', np.float32(img_bin_diff * 0.01))
            dep_bin_diff= np.abs(bg_dep - img_dep)
            dep_bin_diff[~mask]=0
            dep_bin_diff = cv2.medianBlur(np.uint8(dep_bin_diff*500), 3)
            dep_bin_diff = cv2.blur(dep_bin_diff,(3, 3))/500
            #cv2.imshow('dep_bin_diff', np.float32(dep_bin_diff*10))

            final_diff=(img_bin_diff)*IMG_WEIGHT
            final_diff+=dep_bin_diff*DEP_WEIGHT

            final_diff[final_diff < 0.01]=0

            valid_edge_eara=final_diff>VALID_EARA_MIN

            valid_edge_eara = cv2.dilate(np.uint8(valid_edge_eara), kernel_1, iterations=1)
            valid_edge_eara=cv2.erode(np.uint8(valid_edge_eara), kernel_2, iterations=1)
            
            #cv2.imshow('valid_edge_eara', np.float32(valid_edge_eara))
            img_edge = canny(final_diff, EDGE_DET_TH).astype(int)
            img_edge[valid_edge_eara>0]=0
            #cv2.imshow('final_diff',np.float32(final_diff*10))

        #背景反射解决方案
        # edge_mode_mask=bg_amp<255#np.bitwise_and(bg_amp<255,img_amp<300)
        # cv2.imshow('edge_mode_mask', np.float(edge_mode_mask))
        # if EDGE_DEP_BY_AMP:  # 基于强度图的边沿检测
        #     img_amp_diff = np.abs(img_amp - bg_amp)#np.abs(bg_dep - img_dep)*1000+
        #     img_bin_diff = img_amp_diff > EDGE_DEP_AMP_TH
        #     img_bin_diff=np.bitwise_and(img_bin_diff,edge_mode_mask)
        #     dep_bin_diff= np.bitwise_and(mask,~edge_mode_mask)
        #     final_diff=np.bitwise_or(img_bin_diff,dep_bin_diff)
        #     img_edge = canny(final_diff, EDGE_DET_TH).astype(int)
        #     cv2.imshow('final_diff',np.int8(final_diff*100000))

        else:  # 基于深度图的边沿检测
            if GROUND_FILL: img_dep[~mask] = ground_dist  # 无效区域填入平均地面深度，防止产生过多边沿

            img_edge = canny(img_dep, EDGE_DET_TH).astype(int)

        # 伪彩色
        img_dep_cmap = img_to_cmap(img_dep, mask=mask, vmin=VIEW_DMIN, vmax=VIEW_DMAX)

        # 当边界检测足够“干净”时才启动立方体检测（不检测杂乱图像）
        go_on = img_edge.sum() < IMG_EDGE_POINT_MAX
        if not go_on: print('stop detection, img_edge.sum():', img_edge.sum())

        # 矩形检测
        if go_on:
            h, theta, d = hough_line(img_edge)
            go_on = h.max() > HOUGH_LINE_CUM_MIN  # 当检出足够长的边界直线时才继续检测

        if go_on:
            # 提取直线参数
            th = max(HOUGH_LINE_CUM_MIN, HOUGH_LINE_TH_COEFF * h.max())  # 门限，用于直线参数提取
            accum, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=th)
            num0 = len(accum)
            print('%02d ' % num0, end='')
            go_on = len(accum) < LINE_NUM_MAX and len(accum) >= 4  # 当检出足够少的直线时才继续（不检测杂乱图像）
            if not go_on: print('stop detection, len(accum):', len(accum))

        if flag_passed == False:
            pass_count += 1
            if pass_count > 5:
                flag_passed = True

        while go_on:
            # 检出所有矩形
            rect = find_rect_from_hough_points(angles, dists)# TIME
            num1 = len(rect)
            print('%02d ' % num1, end='')
            if num1 == 0: break
            # if True:#len(rect)==0:
            #     def points_to_rect(points):
            #         rect = []
            #         for point in points:
            #             #point = point['Points']
            #             len = np.linalg.norm(point[1:] - np.tile(point[0], [3, 1]), axis=-1)
            #             idx = np.argsort(len)
            #             rect.append(
            #                 (point[0], point[idx[1] + 1], point[idx[0] + 1], point[idx[2] + 1], 0, 10))  # 后面两位随便给的
            #         return rect
            #
            #     def find_coordinates(combined_lines, outputfile=None):
            #         boxes = []
            #         min_height = 10
            #         min_width = 10
            #         res, contours, h = cv2.findContours(combined_lines.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #         for cont in contours:
            #             approx = cv2.approxPolyDP(cont, 0.03 * cv2.arcLength(cont, True), True)
            #             rect_points = {}
            #             if len(approx) == 4:
            #                 x, y, width, height = cv2.boundingRect(cont)
            #                 if (width > min_width and height > min_height):
            #                     rectangle = cv2.minAreaRect(cont)
            #                     box = cv2.boxPoints(rectangle)
            #                     box = np.int0(box)
            #                     #rect_points["Points"] = box
            #                     boxes.append(box)
            #         return boxes
            #     box=find_coordinates(final_diff)
            #     if len(box)>0:
            #         rect=points_to_rect(box)
            #     else:
            #         rect=[]
            # num1 = len(rect)
            # print('%02d ' % num1, end='')
            # if num1 == 0: break
            # 根据长短边尺寸过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if
                    verify_rect_by_size_ratio(n0, n1, n2, n3)]
            num2 = len(rect)
            print('%02d ' % num2, end='')
            if num2 == 0: break

            # 根据面积过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if verify_rect_by_area(n0, n1, n2, n3)]
            num3 = len(rect)
            print('%02d ' % num3, end='')
            if num3 == 0: break

            # 根据边沿图过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if
                    verify_rect_by_edge(1 - img_edge, n0, n1, n2, n3)]
            num4 = len(rect)
            print('num4 %02d ' % num4, end='')
            if num4 == 0: break

            # 太靠近图像边界的矩形过滤
            #rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if edge_rect_removal(n0, n1, n2, n3)]

            # 根据重叠情况合并
            rect = rect_merge(rect)
            num5 = len(rect)
            print('num5 %02d ' % num5, end='')
            if num5 == 0: break

            # 只保留最大面积的盒子
            if RECT_KEEP_MAX: rect = keep_max_area_rect(rect)


            # 计算盒子参数并显示
            c, text_y = 0, 15
            for i in range(len(rect)):
                flag_passed = False
                pass_count = 0
                n0, n1, n2, n3 = rect[i][0], rect[i][1], rect[i][2], rect[i][3]
                # 计算盒子顶面到镜头的距离
                box_dep, box_amp, Confidence = calc_box_dep_amp(img_dep, img_amp, n0, n1, n2, n3, mask)
                #if Confidence<MIN_CONFIDENCE: break#如果矩形内的有效点太少认为结果不可信任
                    
                box_wid, box_len = calc_box_wid_len(n0, n1, n2, n3, box_dep)  # 盒子长宽像3素尺寸
                # 投射到三维空间平面
                if box_wid < box_len: box_wid, box_len = box_len, box_wid
                box_hgt = ground_dist - box_dep  # 计算盒子的高度（扣除地面高度）
                ######## 手动校准 #########
                if MANUAL_CALI == True:
                    #box_wid, box_len, box_hgt = meas_conv.conv(box_wid, box_len, box_hgt)  # 手动校准
                    if ENABLE_USER_CALIB_PARAM:
                        box_hgt-=user_calib_param

                # 追踪质心，并更新矩形运动速度
                mean_point = np.mean([n0, n1, n2, n3], axis=0)
                speed = speed_tk.cal_speed([frame_cnt, mean_point[0], mean_point[1]], box_dep)
                # if (((mean_point[1] - (IMG_HGT / 2))**2 + (mean_point[1]-(IMG_WID / 2))**2 )< 5000 and (speed < 0.5 or box_hgt<0.03)):#防止背景反光干扰
                #     break
                
                # 绘制显示检测得到的矩形
                draw_poly4(img_dep_cmap, n0, n1, n2, n3, thickness=2)
                frame_info = '%d: %03d*%03d*%04d(mm) speed:%03f (m/s)' % (c, box_wid * 1000, box_len * 1000, box_hgt * 1000, speed)
#                if GRAND_DEPTH_CALIBRATE_FROM_FILE  == False :
#                    box_hgt=Ground_Depth-box_hgt
#                    break
                if MEAS_FILTER: box_wid, box_len, box_hgt = meas_filter.filter(box_wid, box_len, box_hgt)
                ave_info = '%d: %03d*%03d*%04d(mm)' % (c, box_wid * 1000, box_len * 1000, box_hgt * 1000)  # 平均后的值
                cv2.putText(img_dep_cmap, frame_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                text_y += 15
                c += 1
                print('\n box_wid:%f ,box_len %f,box_hgt %f'%( box_wid*100,box_len*100,box_hgt*100))
                if MEAS_CONV:
                    fp_log.write('%d,%f,%f,%f\n' % (frame_cnt, box_wid, box_len, box_hgt))
            #print('')
            break

        if flag_passed == True and ave_info != None:
            cv2.putText(img_dep_cmap, ave_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 显示图像
        #viewer.update_pan_img_rgb(img_dep_cmap, pan_id=(1, 0))
        viewer.update_pan_img_rgb(img_to_cmap(1 - img_edge, mask=None, vmin=0, vmax=1), pan_id=(0, 1))
        if img_amp is not None: 
            img_amp_s=img_amp
            img_amp_s[~BACKGROUND_MASK]=0
            #viewer.update_pan_img_gray(img_amp_s * VIEW_AMP_SCALE, pan_id=(0, 1))

        # 屏幕刷新
        viewer.update()
        evt, param = viewer.poll_evt()
        if evt is not None:
            if evt == 'quit': break

        if PLAYBACK_END is not None:
            if frame_cnt == PLAYBACK_END: break

    fp_log.close()

if __name__ == '__main__':
    main()


