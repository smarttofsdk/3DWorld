# -*- coding: utf-8 -*-
from clor import *

# 相机初始化
parms = read_tof_params('tof_params.txt')

dmcam.init(None)
dmcam.log_cfg(dmcam.LOG_LEVEL_INFO,
              dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)

devs = dmcam.dev_list()
if devs is None:
    print(" No device found")
    sys.exit(1)

dev = dmcam.dev_open(None)
# dmcam.cap_set_frame_buffer(dev, None, 320 * 240 * 4 * 10)
dmcam.cap_set_callback_on_frame_ready(dev, None)
dmcam.cap_set_callback_on_error(dev, None)

# show batch param set
print("-> batch param parameters write...\n ")
wparams = {
    dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_HDR_INTG_TIME: dmcam.param_val_u(),
    dmcam.PARAM_FRAME_RATE: dmcam.param_val_u()
}
wparams[dmcam.PARAM_INTG_TIME].intg.intg_us = parms[0]  # 1400
wparams[dmcam.PARAM_HDR_INTG_TIME].intg.intg_us = parms[1]  # 1400
wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = parms[2]  # 10

## Oprerate fileter
# DMCAM_FILTER_ID_AMP
amp_min_val = dmcam.filter_args_u()
amp_min_val.min_amp = parms[3]  # 40
if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val, sys.getsizeof(amp_min_val)):
    print(" set amp to %d %% failed" % 0)
if not dmcam.filter_disable(dev, dmcam.DMCAM_FILTER_ID_MEDIAN):
    print(" disable median filter failed")
hdr = dmcam.filter_args_u()
if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_HDR, hdr, sys.getsizeof(hdr)):
    print(" enable hdr filter failed")

ret = dmcam.param_batch_set(dev, wparams)
assert ret is True

# show batch param get
print("-> batch param parameters reading...\n")
params_to_read = list(range(dmcam.PARAM_ENUM_COUNT))
param_val = dmcam.param_batch_get(dev, params_to_read)
assert param_val is not None
print("dev_mode = %d" % param_val[dmcam.PARAM_DEV_MODE].dev_mode)
print("mode_freq = %d" % param_val[dmcam.PARAM_MOD_FREQ].mod_freq)
print("vendor: %s" % param_val[dmcam.PARAM_INFO_VENDOR].info_vendor)
print("product: %s" % param_val[dmcam.PARAM_INFO_PRODUCT].info_product)
print("max frame info: %d x %d, depth=%d, fps=%d, intg_us=%d"
      % (param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_width,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_height,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_frame_depth,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_fps,
         param_val[dmcam.PARAM_INFO_CAPABILITY].info_capability.max_intg_us))
print([hex(v) for v in param_val[dmcam.PARAM_INFO_SERIAL].info_serial.serial])
print("version: sw:%d, hw:%d, sw2:%d, hw2:%d"
      % (param_val[dmcam.PARAM_INFO_VERSION].info_version.sw_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.hw_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.sw2_ver,
         param_val[dmcam.PARAM_INFO_VERSION].info_version.hw2_ver))
print("frame format = %d" % param_val[dmcam.PARAM_FRAME_FORMAT].frame_format.format)
print("fps = %d" % param_val[dmcam.PARAM_FRAME_RATE].frame_rate.fps)
print("illum_power=%d %%" % param_val[dmcam.PARAM_ILLUM_POWER].illum_power.percent)
print("intg = %d us" % param_val[dmcam.PARAM_INTG_TIME].intg.intg_us)

print("tl:%.2f, tr:%.2f, bl:%.2f, br:%.2f, ib:%.2f\n"
      % (param_val[dmcam.PARAM_TEMP].temp.tl_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.tr_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.bl_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.br_cal / 10,
         param_val[dmcam.PARAM_TEMP].temp.ib_cal / 10
         ))

print(" Start capture ...")
dmcam.cap_start(dev)

# img = np.frombuffer(file_dep.read(2 * 320 * 240 * f_cnt), dtype=np.uint16)
f = bytearray(320 * 240 * 4 * 2)




## 主程序入口
flag = 0
fall_num = 0
fine_num = 0
isFallDown = 0
tflag = 0

count=0
spd = CalMeanSpeed(rx= -30)

foldername=r'fallData'
# alltime = TimeRecord()

bgflag=True
last_center = np.array([0,0])
denoise_kernelo = np.ones((3, 3), np.uint8)
# denoise_kernelo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


while True:
    finfo = dmcam.frame_t()
    ret = dmcam.cap_get_frames(dev, 1, f, finfo)
    # print("get %d frames" % ret)
    if ret > 0:
        w = finfo.frame_info.width
        h = finfo.frame_info.height
        dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
        gray_cnt, gray = dmcam.frame_get_gray(dev, w * h, f, finfo.frame_info)

        if dist_cnt == w * h:

            # -----------------------------------------
            img_dep = (dist.reshape(h, w) * 1000)
            dep0 = img_dep.copy()
            Z = gray.reshape(h, w)
            
            count += 1
            if bgflag :#count < 10:
                dep_bg = img_dep
                cv2.imshow('bg',cv2.convertScaleAbs(img_dep, None, 1 / 16))
            else:
                tofcap.tof_cap(img_dep,Z)

                img_amp = cv2.convertScaleAbs(Z, None, 1)
                img_fall = cv2.convertScaleAbs(img_dep, None, 1 / 16)

                img_move = mog.apply(img_fall)
                img_move = cv2.morphologyEx(img_move, cv2.MORPH_OPEN, denoise_kernelo, iterations=2)
                img_move = cv2.morphologyEx(img_move, cv2.MORPH_CLOSE, denoise_kernelo, iterations=2)
                
                img_fall = cv2.merge([img_fall] * 3)
                img_fall = cv2.applyColorMap(img_fall, cv2.COLORMAP_RAINBOW)
                panl = np.ones_like(img_fall,dtype = 'uint8')*255
                    

                # 去背景
                if dep_bg is not None:
                    img_dep[np.abs(img_dep.copy() - dep_bg) < 100] = 0

                img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                                    amp_th=50,  # 红外图
                                    dmax=4000, dmin=500,  # 深度图
                                    # cutx=40, cuty=10  # 图像四周区域
                                    )
                img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernelo, iterations=2)
                img_dep[img_hand == 0] = 0
                depnb = cv2.convertScaleAbs(img_dep, None, 1 / 16)
                depnb = cv2.merge([depnb] * 3)

                _, markers, stats, centroids = cv2.connectedComponentsWithStats(img_hand)
                area_th = 1500
                detflag = 0
                human_num  = 0

                
                min_d = 20000
                depnb = cv2.convertScaleAbs(img_dep, None, 1 / 16)
                depnb = cv2.merge([depnb] * 3)

                for i in range(1, len(stats)):
                    center = centroids[i].astype('int')
                    dd=transxy(center,dep0)[0][2]
                    human_area = stats[i][4]*dd*dd

                    if human_area > area_th:
                        human_num+=1
                        detflag = 1
                        hum_stat = stats[i]
                        cv2.rectangle(depnb, (hum_stat[0], hum_stat[1]),
                                    (hum_stat[0] + hum_stat[2], hum_stat[1] + hum_stat[3]), color=[0, 255, 0],
                                    thickness=2)
                        d_center = np.sum(np.fabs(last_center - center))
                        if min_d > d_center:
                            min_d = d_center
                            min_id = i
                        
                if detflag:
                    hum_stat = stats[min_id]
                    center = centroids[min_id].astype('int')
                    last_center = center.copy()
                    img_hand[markers != min_id] = 0
                    img_dep[img_hand == 0] = 0
               

                    
                    cv2.rectangle(depnb, (hum_stat[0], hum_stat[1]),
                                    (hum_stat[0] + hum_stat[2], hum_stat[1] + hum_stat[3]), color=[0, 0, 255],
                                    thickness=2)

                    cbk = hum_stat[3] / hum_stat[2]

                    # 速度满足要求
                    vy = spd.meanSpeed(center, dep0)[1]

                    if vy > 0.8:
                        flag = 1

                    if flag == 1 and cbk<1.3: #(hum_stat[3] < 120 or cbk < 0.85):
                        fall_num = fall_num + 1
                    if flag == 1 and fall_num == 3:
                        isFallDown = 1
                        # 发生跌倒，这里加响应
                        #########################

                        # simpleaudio_play('falldown_ring.wav')

                        ##########################
                        fall_num = 0
                        flag = 0

                    # 高度 角度要求
                    # if hum_stat[3] > 180 or cbk > 1.8:
                    if cbk>1.5:
                        fine_num = fine_num + 1
                    if fine_num == 5:
                        flag = 0
                        fine_num = 0
                        isFallDown = 0

                cv2.putText(panl, "Fall Detect: ", (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
                cv2.putText(panl, "Move Detect:", (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
                cv2.putText(panl, "People Number:", (20, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)

                if isFallDown == 1:
                    cv2.putText(panl, "FALL DOWN!", (60, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                    cv2.circle(panl, (40, 50),10, (0, 0, 255), -1)
                else:
                    cv2.putText(panl, "FINE!", (60, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
                    cv2.circle(panl, (40, 50), 10, (0, 255, 0), -1)

                if (img_move>0).sum()>1000:
                    cv2.putText(panl, "Move.", (40, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                else:
                    cv2.putText(panl, "No Move.", (40, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

                if human_num ==0 :
                    cv2.putText(panl, "No Person.", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)
                elif human_num ==1 :
                    cv2.putText(panl, "Single Person.", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                else :
                    cv2.putText(panl, "Multi Person.", (40, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 0, 1)



                # img_fall = cv2.resize(img_fall, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                depnb = cv2.resize(depnb, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


                img_fall = np.concatenate((panl,img_fall),axis = 0)
                cv2.imshow("Status", img_fall)
                cv2.imshow('nobackground', depnb)

            key = cv2.waitKey(10)
            if key == 27:
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord('q'):
                break
            elif key == ord('t'):
                tflag ^= 1
            elif key == ord('b') and bgflag:
                bgflag = False
                tofcap = TofCapture(time.strftime("tof%m%d_%H%M%S"), 0, foldername)

cv2.destroyAllWindows()

