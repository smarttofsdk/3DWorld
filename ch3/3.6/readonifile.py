import dmcam
import sys
import os
import glob
import cv2
cnt = 0
row = 1

for file in glob.glob(r"dmrep_20200407_132641.oni"):
    print("file=",file)

    read_oni_file_path = file

    dmcam.init(None)

    dmcam.log_cfg(dmcam.LOG_LEVEL_INFO, dmcam.LOG_LEVEL_DEBUG, dmcam.LOG_LEVEL_NONE)
    dev = dmcam.dev_open_by_uri(os.fsencode(file))

    print(" Config capture param ..")
    cap_cfg = dmcam.cap_cfg_t()
    cap_cfg.cache_frames_cnt = 10  # frame buffer = 10 frames

    cap_cfg.en_save_dist_u16 = False  # save dist into ONI file: which can be viewed in openni
    cap_cfg.en_save_gray_u16 = False  # save gray into ONI file: which can be viewed in openni
    cap_cfg.en_save_replay = False  # save raw into ONI file:  which can be simulated as DMCAM device
    cap_cfg.fname_replay = os.fsencode("replay_dist.oni")

    dmcam.cap_config_set(dev, cap_cfg)

    print(" Set paramters ...")
    wparams = {
        dmcam.PARAM_INTG_TIME: dmcam.param_val_u(),
        dmcam.PARAM_FRAME_RATE: dmcam.param_val_u(),
    }

    wparams[dmcam.PARAM_FRAME_RATE].frame_rate.fps = 0
    amp_min_val = dmcam.filter_args_u()
    amp_min_val.min_amp = 0


    amp_min_val = dmcam.filter_args_u()
    amp_min_val.min_amp = 10  # 40
    if not dmcam.filter_enable(dev, dmcam.DMCAM_FILTER_ID_AMP, amp_min_val, sys.getsizeof(amp_min_val)):
        print(" set amp to %d %% failed" % 0)
    if not dmcam.filter_disable(dev, dmcam.DMCAM_FILTER_ID_FLYNOISE):
        print(" disable fly noise filter failed")

    if not dmcam.param_batch_set(dev, wparams):
        print(" set parameter failed")

    if not dmcam.param_batch_set(dev, wparams):
        print(" set parameter failed")

    print(" Start capture ...")
    dmcam.cap_start(dev)

    f = bytearray(640 * 480 * 4 * 2)
    run = True

    frame_buff = []
    try:
        while run:
            # get one frame
            finfo = dmcam.frame_t()
            ret = dmcam.cap_get_frames(dev, 1, f, finfo)

            if ret > 0:
                print(" frame @ %d, %dx%d (%d)" % (
                finfo.frame_info.frame_idx, finfo.frame_info.width, finfo.frame_info.height, finfo.frame_info.frame_size))

                w, h = (finfo.frame_info.width, finfo.frame_info.height)
                gray_cnt = 0

                dist_cnt, dist = dmcam.frame_get_distance(dev, w * h, f, finfo.frame_info)
                gray_cnt, gray = dmcam.frame_get_gray(dev, w * h, f, finfo.frame_info)

                img_dep = (dist.reshape(h, w) * 1000)
                img_amp = gray.reshape(h, w)


                if dist_cnt != w * h:
                    dist = None
                if gray is not None:
                    img_amp = cv2.convertScaleAbs(img_amp, None, 1 / 2)

                    cv2.imshow('gray', img_amp)

                if dist is not None:
                    img_dep = cv2.convertScaleAbs(img_dep, None, 1 / 8)
                    cv2.imshow('dep', img_dep)
                    cv2.waitKey(10)

            else:
                break
    except:
        # workbook.close()
        print("finish file %s", read_oni_file_path)
