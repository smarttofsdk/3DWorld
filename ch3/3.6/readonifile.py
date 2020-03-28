##  功能描述：
#   利用dmcam库读取ONI文件
#   输入参数：
#   FNAME: ONI文件路径和名称
#   CAM_WID,CAM_HGT: 相机参数

import numpy as np
import import dmcam

FNAME=b'./sony_data/sony_model.oni'
CAM_WID=640
CAM_HGT=480

dmcam.init(None)
dmcam.log_cfg(dmcam.LOG_LEVEL_INFO,dmcam.LOG_LEVEL_DEBUG,dmcam.LOG_LEVEL_NONE)

cam = dmcam.dev_open_by_uri(FNAME)

filter_arg=dmcam.filter_args_u()
filter_arg.offset_mm=DMCAM_OFFSET
dmcam.filter_enable(cam,dmcam.DMCAM_FILTER_ID_OFFSET,filter_arg,sys.getsizeof(filter_arg))
dmcam.cap_start(cam)
dmcam.cap_set_frame_buffer(cam, None, DMCAM_BUF_SZ*CAM_WID*CAM_ HGT*4*2)

frame_data = bytearray(CAM_WID*CAM_HGT*4*4)
frame_dist = [np.zeros((CAM_HGT,CAM_WID))]
frame_gray = [np.zeros((CAM_HGT,CAM_WID))]
frame_cnt=0

while True:
    finfo = dmcam.frame_t()
    ret=dmcam.cap_get_frame(cam, frame_data, finfo)
    if ret>0:
        , frame_dist = dmcam.frame_get_distance(cam, CAM_WID*CAM_ HGT, frame_data, finfo.frame_info)
        , frame_gray = dmcam.frame_get_gray    (cam, CAM_WID*CAM_HGT, frame_data, finfo.frame_info)
        frame_cnt+=1

    else:
        time.sleep(1.0/float(DMCAM_FRAMERATE))
        continue
