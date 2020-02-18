# import zmq
# import struct
import os
# import time
import cv2
import sys
import numpy as np
from hand_segment import pre_filter, update_contour, circle_area_mean_shift, find_init_search_point
from sklearn.externals import joblib
from xy_calc_tip import xy_calc_tip
from datetime import datetime

if getattr(sys, 'frozen', False):
    res_path = sys._MEIPASS
else:
    res_path = os.path.split(os.path.realpath(__file__))[0]
# svm_persist_file = "%s/model/wwh_20180413-1.svm" % res_path
# pca_persist_file = "%s/model/wwh_20180413-1.pca" % res_path
svm_persist_file = "%s/model/20190922.svm" % res_path
pca_persist_file = "%s/model/20190922.pca" % res_path
svc = joblib.load(svm_persist_file)
pca = joblib.load(pca_persist_file)

label_list = ("big5", "bravo", "fist", "none", "vict")
key_dir_map = {
    '0': "big5",
    '1': "fist",
    '2': "none",
    '3': "vict",
    '4': "bravo",
}

def pre_filter(img):
    if img.dtype == np.uint16:
        img_f = img.astype(np.float32)
        ret, img_f = cv2.threshold(img_f, 0, 255, cv2.THRESH_BINARY_INV)
        img_src = img_f.astype(np.uint8)
    else:
        img_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img_src = cv2.threshold(img_src, 35, 255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret, img_src = cv2.threshold(img_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ----------- filer section --------------
    # img_src = cv2.adaptiveThreshold(img_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    img_src_f = img_src
    img_src_f = cv2.bitwise_not(img_src_f)
    # img_src_f = cv2.GaussianBlur(img_src_f, (3, 3), 0.2)

    img_src_f = cv2.erode(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.dilate(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.dilate(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.erode(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # cv2.imshow("p0", img_src_f)
    # cv2.waitKey(1)

    # find max contour and fill hole
    i, contour, h = cv2.findContours(img_src_f, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    max_area = 0
    max_contour = None
    for c in contour:
        cur_area = cv2.contourArea(c)
        if max_area < cur_area:
            max_area = cur_area
            max_contour = c

    # print(max_area)
    if max_area < 1000:
        return img_src_f, None

    for c in contour:
        if not np.array_equal(c, max_contour):
            # use contours to fill hole
            cv2.drawContours(img_src_f, [c], 0, 255, -1)
    if max_contour is not None and len(max_contour) > 0:
        max_contour = cv2.approxPolyDP(max_contour, 1.5, True)
        print(" hand contour len=%d" % len(max_contour))

    return img_src_f, max_contour

def preprocess_py_v2(img):
    global img_pp
    # pre filter the image
    img_src_f, max_contour = pre_filter(img)

    if max_contour is None or len(max_contour) < 3:
        print("none")
        return None

    img_pp = cv2.cvtColor(img_src_f, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("pre", img_pp)
    img_bound_wh = (img_src_f.shape[1], img_src_f.shape[0])
    # --- calc palm center
    init_point, init_r, bstatus = find_init_search_point(max_contour, img_bound_wh)

    if init_point is None:
        print(" invalid pos!")
        return None
    # init_point = centroid
    # init_r = 2.0

    c, r = circle_area_mean_shift(max_contour, init_point, init_r=init_r,
                                  min_converge=1.0, area_ratio_threshold=0.6, radius_step=2.0, img_dump=img_pp)

    # calc updated contour
    new_contour = update_contour(max_contour, c, r, img_bound_wh, init_point)
    hull_points = cv2.convexHull(new_contour)

    # ----- visualization to img_dump
    # draw contour/ palm circle and centroid
    # contour_moment = cv2.moments(max_contour)
    # centroid = (int(contour_moment["m10"] / contour_moment["m00"]),
    #             int(contour_moment["m01"] / contour_moment["m00"]))
    # cv2.circle(img_pp, c, int(r), (255, 128, 128), 1)  # parm circle
    # cv2.drawContours(img_pp, [max_contour], 0, (255, 128, 0), 2, 4)
    # cv2.circle(img_pp, centroid, 1, (128, 128, 0), 1)

    # draw convexHull
    # if max_contour is not None and len(max_contour) > 0:
    #     cv2.drawContours(img_pp, [hull_points], 0, (128, 255, 0), 2, 4)
    # cv2.drawContours(img_pp, [new_contour], 0, (50, 50, 255), 2)
    # cv2.imshow("fin", img_pp)

    # # save to file
    # fpath, fname = os.path.split(filename)
    # os.mkdir("%s/preprocess" % fpath) if not os.path.isdir("%s/preprocess" % fpath) else None
    # cv2.imwrite("%s/preprocess/%s" % (fpath, fname), img_dump)

    return np.concatenate((cv2.HuMoments(cv2.moments(new_contour)).flatten(),
                           cv2.HuMoments(cv2.moments(hull_points)).flatten()))

def preprocess(img):
    img_src_f, max_contour = pre_filter(img)

    if max_contour is None or len(max_contour) < 3:
        return None

    img_pp = cv2.cvtColor(img_src_f, cv2.COLOR_GRAY2BGR)

    img_bound_wh = (img_src_f.shape[1], img_src_f.shape[0])
    # --- calc palm center
    init_point, init_r, bstatus = find_init_search_point(max_contour, img_bound_wh)

    if init_point is None:
        print(" invalid pos!")
        return None
    # init_point = centroid
    # init_r = 2.0
    # print(init_point, init_r)
    c, r = circle_area_mean_shift(max_contour, init_point, init_r=init_r,
                                  min_converge=1.0, area_ratio_threshold=0.6, radius_step=2.0, img_dump=img_pp)
    # print(c, r)

    # calc updated contour
    new_contour = update_contour(max_contour, c, r, img_bound_wh, init_point)
    hull_points = cv2.convexHull(new_contour)

    # ----- visualization to img_dump
    # draw contour/ palm circle and centroid
    contour_moment = cv2.moments(max_contour)
    centroid = (int(contour_moment["m10"] / contour_moment["m00"]),
                int(contour_moment["m01"] / contour_moment["m00"]))
    cv2.circle(img_pp, c, int(r), (255, 128, 128), 1)  # parm circle
    cv2.drawContours(img_pp, [max_contour], 0, (255, 128, 0), 2, 4)
    cv2.circle(img_pp, centroid, 1, (128, 128, 0), 1)

    # draw convexHull
    hull = cv2.convexHull(max_contour, False)
    if max_contour is not None and len(max_contour) > 0:
        cv2.drawContours(img_pp, [hull], 0, (128, 255, 0), 2, 4)
    cv2.drawContours(img_pp, [new_contour], 0, (50, 50, 255), 2)
    # cv2.imshow('img_pp',img_pp)
    # cv2.waitKey(1)
    #
    # save to file
    # fpath, fname = os.path.split(file)
    # os.mkdir("%s/preprocess" % fpath) if not os.path.isdir("%s/preprocess" % fpath) else None
    # cv2.imwrite("%s/preprocess/%s" % (fpath, fname), img_pp)

    return np.concatenate((cv2.HuMoments(cv2.moments(new_contour)).flatten(),
                           cv2.HuMoments(cv2.moments(hull_points)).flatten()))

def reset():
    big5_cnt = 0
    fist_cnt = 0
    vict_cnt = 0
    bravo_cnt = 0
    # flag = 0
    print("reset finish!")
    return big5_cnt, fist_cnt, vict_cnt, bravo_cnt  #, flag

pr = 0.85

def static(img):
    cv2.flip(img, -1, img)
    img = np.rot90(img)
    img = np.rot90(img)
    # img_room = (img.copy() / 16).astype(np.uint8)
    # cv2.imshow("scene", img_room)
    # img[img>3000] = 0
    #
    # ## 深度切割阈值确定
    # arr = img.flatten()
    # arr1 = arr[arr > 30]
    # arr1.sort()
    # aver = sum(arr1[:800]) / 700
    #
    # if aver > 2100:
    #     dep_fil = 2100
    #
    # # print("区域的深度平均值=", aver)
    # img[img > aver + 5] = 0
    # mask = np.bitwise_and(img > aver, img < 65535)

    img = np.where(img < (max(np.min(img),380) + 200), img, 0)
    mask = np.where(img,1,0)
    mask[img<max(np.min(img),380)]=0
    mask = mask.astype(np.uint8)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # cv2.imshow("mask",(mask*255).astype(np.uint8))
    # cv2.waitKey(1)

    img = img*mask

    #  # 丢弃深度大于aver的图像信息
    # img = cv2.medianBlur(img, 3)
    # img_u8 = (img.copy()/16).astype(np.uint8)
    #
    # ret, bin_img = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY)
    # # cv2.imshow("bin_img",bin_img)
    # # cv2.waitKey(1)
    #
    # ## 最大联通区域和空白填充
    # i, contour, h = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # max_area = 0
    # max_contour = None
    # for c in contour:
    #     cur_area = cv2.contourArea(c)
    #     if max_area < cur_area:
    #         max_area = cur_area
    #         max_contour = c
    # c_min = []
    # for c in contour:
    #     area = cv2.contourArea(c)
    #     if area < max_area:  # 3500:
    #         c_min.append(c)
    #         cv2.drawContours(bin_img, c_min, -1, 0, thickness=-1)
    # c_min.clear()
    # mask = np.bitwise_and(bin_img > 0, bin_img == 255)
    # mask_img = img_u8 * mask
    # mask_img = cv2.medianBlur(mask_img, 3)
    # # cv2.imshow('mask_img',mask_img)
    # # cv2.waitKey(1)
    #
    # fvector = preprocess_py_v2(mask_img)
    fvector = preprocess(img)

    if fvector is None:
        pose_type = -1
        result = 0
        result_str = pose_type
    else:
        pca_vector = pca.transform([fvector])
        pose_type = svc.predict(pca_vector)
        # result_str = pose_type[0]
        if pose_type:
            ## SVM分类为的静态手势概率
            pro = svc.predict_proba(pca_vector)
            print("pro = ", pro[0])
            proba = max(pro[0])
            max_idx = np.argmax(pro[0])
            ## 分类置信率阈值
            if proba >= pr and max_idx == 0:
                result = 1
                result_str = '五指张开'
            elif proba >= pr and max_idx == 3:
                result = 2
                result_str = 'V'
            elif proba >= pr and max_idx == 2:
                result = 3
                result_str = '拳头'
            elif proba >= pr and max_idx == 1:
                result = 4
                result_str = '大拇指'
            else:
                result = 0
                result_str = '无动作'

            # print('result:',result)
            # print(proba)
            # print('-----------------------------------------')
    # cv2.imshow("res", mask_img)
    # cv2.waitKey(0)

    return result, result_str

