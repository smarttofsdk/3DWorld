import cv2
import numpy as np
from global_cfg import *


def xy_calc_tip(new_contour, c, r):

    # select by point angle
    finger_tip_list, bend_list = [], []
    num = int(new_contour.size / 2)
    FTIP_TRK_BEND_NUM=int(num/20)
    for k in range(num):
        x_backward, y_backward = new_contour[(k - FTIP_TRK_BEND_NUM) % num][0][0], new_contour[(k - FTIP_TRK_BEND_NUM) % num][0][1]
        x_forward, y_forward   = new_contour[(k + FTIP_TRK_BEND_NUM) % num][0][0], new_contour[(k + FTIP_TRK_BEND_NUM) % num][0][1]
        bend = calc_point_angle(new_contour[k][0][0], new_contour[k][0][1],
                                x_backward, y_backward, x_forward, y_forward)
        if bend > FTIP_TRK_BEND_TH2:
            finger_tip_list.append(new_contour[k][0])
            bend_list.append(bend)
            # cv2.circle(img_dump, (new_contour[k][0][0],new_contour[k][0][1]), 5, (255, 255, 128), 1)

    # calc finger tip
    max_d = r
    min_bend = 0
    finger_tip = c
    for (p, b) in zip(finger_tip_list, bend_list):
        tmp_dist = np.linalg.norm(p - np.array(c))
        # if tmp_dist <= r:
        #     continue
        if tmp_dist > max_d:
            finger_tip = (p[0], p[1])
            max_d = tmp_dist
            min_bend = b

    # q0, q1
    if finger_tip == c:
        q0 = 0
        q1 = 0
    else:
        q0 = (max_d - r) / r
        q1 = min_bend

    return finger_tip, [q0, q1]


# 计算三点角度
def calc_point_angle(x, y, xa, ya, xb, yb):

    vx1, vy1 = x-xa, y-ya
    vx2, vy2 = x-xb, y-yb
    tmp = np.sqrt(float((vx1**2 + vy1**2) * (vx2**2 + vy2**2)))
    if tmp == 0:
        tmp = 1.0 # avoid div-0 error
    return float(vx1 * vx2 + vy1 * vy2) / tmp
