#!/usr/bin/python3
# coding=utf-8

## 立方体测量

import itertools    as itt
import sys

from   skimage.feature import canny
from   skimage.transform import (hough_line, hough_line_peaks)

sys.path.append('./')
from geo_tools         import *
from depth_img_view_cv import *

if not PLAYBACK:
    from dmcam_dev import *


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


## 矩形匹配
def match_rect(n0, n1, n2, n3, n4, n5, n6, n7, th=RECT_MATH_TH ** 2):
    # 先通过中心位置快速发现不匹配的情况
    c0 = calc_rect_cent(n0, n1, n2, n3)
    c1 = calc_rect_cent(n4, n5, n6, n7)
    dc = calc_point_dist_sqr(c0, c1)
    if dc > th: return False, n0, n1, n2, n3

    # 对第一个矩形的4个顶点n0~n3和第二个矩形的顶点n4~n7进行两两配对
    nsel = [n0, n1, n2, n3]

    # n4顶点匹配
    d0 = calc_point_dist_sqr(n0, n4)
    d1 = calc_point_dist_sqr(n1, n4)
    d2 = calc_point_dist_sqr(n2, n4)
    d3 = calc_point_dist_sqr(n3, n4)
    m0 = nsel[np.argmin([d0, d1, d2, d3])]
    if min(d0, d1, d2, d3) > th: return False, n0, n1, n2, n3

    # n5顶点匹配
    d0 = calc_point_dist_sqr(n0, n5)
    d1 = calc_point_dist_sqr(n1, n5)
    d2 = calc_point_dist_sqr(n2, n5)
    d3 = calc_point_dist_sqr(n3, n5)
    m1 = nsel[np.argmin([d0, d1, d2, d3])]
    if min(d0, d1, d2, d3) > th: return False, n0, n1, n2, n3

    # n6顶点匹配
    d0 = calc_point_dist_sqr(n0, n6)
    d1 = calc_point_dist_sqr(n1, n6)
    d2 = calc_point_dist_sqr(n2, n6)
    d3 = calc_point_dist_sqr(n3, n6)
    m2 = nsel[np.argmin([d0, d1, d2, d3])]
    if min(d0, d1, d2, d3) > th: return False, n0, n1, n2, n3

    # n7顶点匹配
    d0 = calc_point_dist_sqr(n0, n7)
    d1 = calc_point_dist_sqr(n1, n7)
    d2 = calc_point_dist_sqr(n2, n7)
    d3 = calc_point_dist_sqr(n3, n7)
    m3 = nsel[np.argmin([d0, d1, d2, d3])]
    if min(d0, d1, d2, d3) > th: return False, n0, n1, n2, n3

    return True, m0, m1, m2, m3


## 矩形跟踪器
class rect_tracker_c:
    def __init__(self):
        # 每个元素内部结构为：(m0,m1,m2,m3,t)，对应矩形均值和最后更新时间
        self.rect_seq = []
        return

    ## 匹配并加入新的矩形
    def match_add(self, n0, n1, n2, n3, t):
        match_flag = False
        for i in range(len(self.rect_seq)):
            m0, m1, m2, m3, t_last, cnt_last = self.rect_seq[i]  # 遍历历史跟踪矩形
            flag, m0, m1, m2, m3 = match_rect(m0, m1, m2, m3, n0, n1, n2, n3)
            if flag:  # 匹配完成
                if MEAS_CONV:
                    w = RECT_TRK_ALPHA ** (t - t_last)  # 跟踪内容融合
                else:
                    w = 0.1 ** (t - t_last)  # 调试时不做滤波
                r0 = m0[0] * w + (1.0 - w) * n0[0], m0[1] * w + (1.0 - w) * n0[1]
                r1 = m1[0] * w + (1.0 - w) * n1[0], m1[1] * w + (1.0 - w) * n1[1]
                r2 = m2[0] * w + (1.0 - w) * n2[0], m2[1] * w + (1.0 - w) * n2[1]
                r3 = m3[0] * w + (1.0 - w) * n3[0], m3[1] * w + (1.0 - w) * n3[1]
                self.rect_seq[i] = (r0, r1, r2, r3, t, cnt_last + 1)  # 更新跟踪矩形
                match_flag = True
                break

        # 如果是新的矩形，加入跟踪
        if not match_flag:
            self.rect_seq.append((n0, n1, n2, n3, t, 1))
        return match_flag

    # 过滤跟踪的矩形列表，去除太老的矩形
    def clean_up(self, t):
        self.rect_seq = [s for s in self.rect_seq if t - s[4] < RECT_TRK_T_MAX]


## 功能描述
#   从hough直线检测得到的参数查找矩形
# 输入参数
#   angles:     检测得到的各个直线的角度
#   dists:      检测得到的各个直线到原点的距离(根据角度值，可以为负数)
#   angle_th:   角度领域范围，用于查找相同角度的直线参数
#   cent_th:    距离领域范围，用于检测矩形的两对对边中心到远点距离是否相同
def find_rect_from_hough_points(angles, dists, angle_th=deg_to_rad(8)):
    num = len(angles)
    rect = []  # 存放结果

    # 所有hough参数转成直线两点的坐标
    lines = {i: hough_param_to_line(angles[i], dists[i]) for i in range(num)}

    # 计算任意两条直线的交点和夹角
    cross_point = [[0] * num for _ in range(num)]
    cross_angle = np.zeros((num, num))
    for i0, i1 in itt.combinations(range(num), 2):
        n0, n1 = lines[i0]
        n2, n3 = lines[i1]
        p = calc_line_cross(n0, n1, n2, n3, cond_max=100)
        a = calc_line_cross_angle(n0, n1, n2, n3)
        cross_point[i0][i1] = cross_point[i1][i0] = p
        cross_angle[i0, i1] = cross_angle[i1, i0] = abs(a)

    # 对任意一条直线，找到和它平行和垂直的线集合
    for i in range(num - 3):
        par_line = [n for n in range(i + 1, num) if cross_angle[i, n] < angle_th]
        org_line = [n for n in range(num) if abs(cross_angle[i, n] - math.pi * 0.5) < angle_th]
        if len(par_line) < 1 or len(org_line) < 2: continue  # 找不到足够的线构成四边形

        # 得到构成矩形的4条直线,计算出矩形的4个交点
        for i1 in par_line:
            for i2, i3 in itt.combinations(org_line, 2):
                c0 = cross_point[i][i2]  # 四个矩形的角点
                c1 = cross_point[i][i3]
                c2 = cross_point[i1][i2]
                c3 = cross_point[i1][i3]

                q0 = abs(math.cos(cross_angle[i, i2]))  # 内角余弦，用于计算矩形畸变量
                q1 = abs(math.cos(cross_angle[i, i2]))
                q2 = abs(math.cos(cross_angle[i, i2]))
                q3 = abs(math.cos(cross_angle[i, i2]))
                q = q0 + q1 + q2 + q3

                r = calc_4point_area(n0, n1, n3, n2)  # 面积

                rect.append((c0, c1, c2, c3, q, r))
    return rect


## 在图片上绘制凸四边形
def draw_poly4(img, n0, n1, n2, n3, color=(255, 255, 255), thickness=3):
    cv2.line(img, (int(n0[0]), int(n0[1])), (int(n1[0]), int(n1[1])), color, thickness)
    cv2.line(img, (int(n2[0]), int(n2[1])), (int(n3[0]), int(n3[1])), color, thickness)
    cv2.line(img, (int(n0[0]), int(n0[1])), (int(n2[0]), int(n2[1])), color, thickness)
    cv2.line(img, (int(n1[0]), int(n1[1])), (int(n3[0]), int(n3[1])), color, thickness)
    return


## 在单色图上根据直线采样点采样
# s list，存放采样点坐标
# 返回图上的采样值列表
def get_img_sample(img_gray, s):
    ymax, xmax = img_gray.shape
    return np.array(
        [img_gray[max(0, min(ymax - 1, int(round(y)))), max(0, min(xmax - 1, int(round(x))))] for x, y in s])


## 功能描述
#   检验n0,n1,n2,n3构成的四边形是否和边沿图对应
#   通过计算直线的边到边沿图中最近距离验证
# 输入参数：
#   img_zero_edge   边沿图，边沿用0对应
#   n0,n1,n2,n3     矩形的四个顶点
#   th              形边到边沿图上边沿线的距离门限
# 输出
#   True/False
def verify_rect_by_edge(img_zero_edge, n0, n1, n2, n3, th=RECT_EDGE_DIST_MAX):
    img_dt = cv2.distanceTransform(img_zero_edge.astype(np.uint8), cv2.DIST_L1, maskSize=3)

    pts, num = gen_line_sample(n0, n1);
    e1 = get_img_sample(img_dt, pts);
    e1 = e1.max() if num < RECT_EDGE_CUT * 2 + 1 else e1[RECT_EDGE_CUT:num - RECT_EDGE_CUT].max()
    pts, num = gen_line_sample(n2, n3);
    e2 = get_img_sample(img_dt, pts);
    e2 = e2.max() if num < RECT_EDGE_CUT * 2 + 1 else e2[RECT_EDGE_CUT:num - RECT_EDGE_CUT].max()
    pts, num = gen_line_sample(n0, n2);
    e3 = get_img_sample(img_dt, pts);
    e3 = e3.max() if num < RECT_EDGE_CUT * 2 + 1 else e3[RECT_EDGE_CUT:num - RECT_EDGE_CUT].max()
    pts, num = gen_line_sample(n1, n3);
    e4 = get_img_sample(img_dt, pts);
    e4 = e4.max() if num < RECT_EDGE_CUT * 2 + 1 else e4[RECT_EDGE_CUT:num - RECT_EDGE_CUT].max()
    return max((e1, e2, e3, e4)) < th


## 功能描述
#   根据长短边比例检验n0,n1,n2,n3构成的四边形是否为矩形
# 输入参数：
#   n0,n1,n2,n3     矩形的四个顶点
#   th              验证矩形的长短边比例门限
# 输出
#   True/False
def verify_rect_by_size_ratio(n0, n1, n2, n3, th=RECT_SIZE_RATIO_MAX):
    d0 = (n0[0] - n1[0]) ** 2 + (n0[1] - n1[1]) ** 2
    d1 = (n2[0] - n3[0]) ** 2 + (n2[1] - n3[1]) ** 2
    d2 = (n0[0] - n2[0]) ** 2 + (n0[1] - n2[1]) ** 2
    d3 = (n1[0] - n3[0]) ** 2 + (n1[1] - n3[1]) ** 2
    return min([d0, d1, d2, d3]) * th > max([d0, d1, d2, d3])


## 功能描述
#   计算四边形中心
# 输入参数
#   n0,n1,n2,n3 四边形四个顶点的xy坐标
# 输出
#   中心的xy坐标
def calc_rect_cent(n0, n1, n2, n3):
    return (n0[0] + n1[0] + n2[0] + n3[0]) / 4.0, (n0[1] + n1[1] + n2[1] + n3[1]) / 4.0


## 功能描述
#   检验n0,n1,n2,n3构成的四边形面积是否足够大
# 输入参数：
#   n0,n1,n2,n3     矩形的四个顶点
#   th              验证矩形的面积门限
# 输出
#   True/False
def verify_rect_by_area(n0, n1, n2, n3, th=RECT_AREA_MIN): return calc_4point_area(n0, n1, n3, n2) > th


## 在图片上绘制填充的凸四边形
def draw_filled_poly4(img, n0, n1, n2, n3, color=(255, 255, 255)):
    points = np.array([[n0[0], n0[1]],
                       [n1[0], n1[1]],
                       [n3[0], n3[1]],
                       [n2[0], n2[1]]], dtype=np.int32)
    cv2.fillConvexPoly(img, points, color)
    return


## 根据深度图计算矩形高度
def calc_box_dep_amp(img_dep, img_amp, n0, n1, n2, n3, mask=None, erode_ker=KER_ERODE):
    img_sel = np.zeros_like(img_dep, dtype=np.uint8)
    draw_filled_poly4(img_sel, n0, n1, n2, n3, color=1)
    if mask is not None:
        img_sel[~mask] = 0
    if erode_ker is not None:  # 面积计算区域的缩小，减少边界误差
        img_sel = cv2.erode(img_sel.astype(np.uint8), erode_ker)
    a = img_sel.sum()
    # valid_dep=np.sort(img_dep[img_sel>0])
    dep_median = np.median(img_dep[img_sel > 0])
    dep = 0 if a == 0 else img_dep[
                               img_sel > 0].sum() / a  # np.mean(np.clip(img_dep[img_sel>0],dep_median-0.1,dep_median+0.1))
    amp = 0 if a == 0 else img_amp[img_sel > 0].sum() / a
    return dep, amp


def pixel_to_pc(pixel, fx, fy, cx, cy, dep=None, coff_x=None, coff_y=None):
    hgt, wid = 240, 320
    if coff_x is None:
        x = np.arange(wid) - cx
        kx = 1.0 / fx
        coff_x = x * kx

    if coff_y is None:
        y = np.arange(hgt) - cy
        ky = 1.0 / fy
        coff_y = y * ky

    pcx = dep * coff_x[pixel[:, 0]]
    pcy = dep * coff_y[pixel[:, 1]]
    return np.concatenate([np.expand_dims(pcx, -1), np.expand_dims(pcy, -1)], -1)
    # pc = np.hstack((pcx.reshape(hgt * wid, 1), pcy.reshape(hgt * wid, 1), img_dep.reshape(hgt * wid, 1)))


## 计算矩形的长宽
def calc_box_wid_len(n0,n1,n2,n3,dep):
    if DEP_CORR==True:
        dep=dep*DEP_CORR_K[0]+DEP_CORR_K[1]
    # 投射到三维空间平面后再计算距离
    pixel=np.array([n0,n1,n2,n3]).astype(np.int)
    position = pixel_to_pc(pixel, MEAS_FX, MEAS_FY, MEAS_CX, MEAS_CY,dep=dep)
    n0, n1, n2, n3=position[0],position[1],position[2],position[3]
    d0=calc_point_dist(n0,n1)
    d1=calc_point_dist(n2,n3)
    d2=calc_point_dist(n0,n2)
    d3=calc_point_dist(n1,n3)
    return (d0+d1)/2.0,(d2+d3)/2.0


## 矩形合并
def rect_merge(rect):
    num = len(rect)

    # 计算所有矩形的中心
    cent = {i: calc_rect_cent(rect[i][0], rect[i][1], rect[i][2], rect[i][3]) for i in range(num)}

    keep = [True] * num
    # 根据矩形中心的相互包容关系合并
    for i, j in itt.combinations(range(num), 2):
        if keep[i] and keep[j]:
            ni0, ni1, ni2, ni3, _, ri = rect[i]
            nj0, nj1, nj2, nj3, _, rj = rect[j]
            cent_i, cent_j = cent[i], cent[j]
            corner_i = np.array([ni0, ni1, ni3, ni2]).astype(int)
            corner_j = np.array([nj0, nj1, nj3, nj2]).astype(int)

            # 矩形i和j重合
            if cv2.pointPolygonTest(corner_i, cent_j, False) > 0 or \
                            cv2.pointPolygonTest(corner_j, cent_i, False) > 0:
                if ri < rj:  # 保留面积大的
                    keep[i] = False
                else:
                    keep[j] = False
    return [rect[i] for i in range(num) if keep[i]]


## 深度图转成点云，经过点云变换后再映射回深度图
def img_dep_trans_remap(img_dep, trans_mat, mask=None):
    pc = img_dep_to_pc(img_dep, MEAS_FX, MEAS_FY, MEAS_CX, MEAS_CY, mask)
    pc_new = pc_trans(trans_mat, pc)
    return pc_to_img_dep(pc_new, MEAS_FX, MEAS_FY, MEAS_CX, MEAS_CY, IMG_WID, IMG_HGT)


## 绘制图像，并在上面加上检出的矩形
def plot_img_rect(img, rect):
    if img is not None: plt.imshow(img)
    for n0, n1, n2, n3, *_ in rect:
        plt.plot((n0[0], n1[0]), (n0[1], n1[1]), 'w')
        plt.plot((n2[0], n3[0]), (n2[1], n3[1]), 'w')
        plt.plot((n0[0], n2[0]), (n0[1], n2[1]), 'w')
        plt.plot((n1[0], n3[0]), (n1[1], n3[1]), 'w')


def box_dep_to_hgt(dep, amp):
    if MEAS_HGT_CORR == 1:
        return MEAS_HGT_PARAM_A * dep + MEAS_HGT_PARAM_B
    elif MEAS_HGT_CORR == 2:
        return MEAS_HGT_PARAM_A * (dep ** 2) + MEAS_HGT_PARAM_B * dep + MEAS_HGT_PARAM_C
    elif MEAS_HGT_CORR == 3:
        return MEAS_HGT_PARAM_A * dep + MEAS_HGT_PARAM_B * amp + MEAS_HGT_PARAM_C
    elif MEAS_HGT_CORR == 4:
        return MEAS_HGT_PARAM_A * (dep ** 2) + MEAS_HGT_PARAM_B * dep + MEAS_HGT_PARAM_C * amp + MEAS_HGT_PARAM_D
    elif MEAS_HGT_CORR == 5:
        return MEAS_HGT_PARAM_A * (dep ** 2) + MEAS_HGT_PARAM_B * dep + MEAS_HGT_PARAM_C * (
        amp ** 2) + MEAS_HGT_PARAM_D * amp + MEAS_HGT_PARAM_E
    elif MEAS_HGT_CORR == 8:
        if dep < 1.6:
            return MEAS_HGT_PARAM_A1 * dep + MEAS_HGT_PARAM_B1
        else:
            return MEAS_HGT_PARAM_A2 * dep + MEAS_HGT_PARAM_B2


## 立方体长宽尺寸的修正
def box_wid_len_corr(w, l, dep, amp):
    if MEAS_WID_LEN_CORR == 0:
        return w * dep / MEAS_F, l * dep / MEAS_F
    elif MEAS_WID_LEN_CORR == 1:
        return w * (dep - amp * 6e-5) / MEAS_F, l * (dep - amp * 6e-5) / MEAS_F
    elif MEAS_WID_LEN_CORR == 2:
        wout = MEAS_WID_LEN_PARAM_A * w * dep ** 2 / MEAS_F + \
               MEAS_WID_LEN_PARAM_B * w * dep / MEAS_F + \
               MEAS_WID_LEN_PARAM_C * w / MEAS_F
        lout = MEAS_WID_LEN_PARAM_A * l * dep ** 2 / MEAS_F + \
               MEAS_WID_LEN_PARAM_B * l * dep / MEAS_F + \
               MEAS_WID_LEN_PARAM_C * l / MEAS_F
        print('w:', w, 'l:', l, 'wout:', wout, 'lout:', lout)
        return wout, lout


## 背景和地面测量
def background_det(cam, viewer):
    print('background detection...')
    bg_dep, bg_amp = calc_bg(cam, viewer, cum_cnt=BACKGROUND_CUM, skip=BACKGROUND_SKIP)
    bg_dep.shape = IMG_HGT, IMG_WID
    bg_amp.shape = IMG_HGT, IMG_WID
    if BACKGROUND_MID_FILTER > 0:
        bg_dep = cv2.medianBlur(bg_dep, BACKGROUND_MID_FILTER)  # 对检测得到的背景做平滑滤波.
    if MEAS_UNDISTORT:
        bg_dep = cv2.undistort(bg_dep, MEAS_DMAT, MEAS_DVEC)
        bg_amp = cv2.undistort(bg_amp, MEAS_DMAT, MEAS_DVEC)
    if MEAS_DEP_TO_Z:
        bg_dep = img_dep_to_z(bg_dep, MEAS_F, MEAS_CX, MEAS_CY)
    print('end of background detection')

    # 地面倾角矫正
    if GROUND_CORR:
        ground_mask = GROUND_MASK  # 地面中心区域
        pc = img_dep_to_pc(bg_dep, MEAS_FX, MEAS_FY, MEAS_CX, MEAS_CY, ground_mask)  # 将地面中心区域变成点云
        nx, ny, nz, *_ = fit_plane_param(pc)  # 拟合平面，计算参数(法向量)
        if nz < 0: nx, ny, nz = -nx, -ny, -nz  # 法向量尽量指向z轴正方向
        a = math.atan(nx / nz)  # 根据现有地面法向量计算矫正角度
        b = math.atan(ny / math.sqrt(nx ** 2 + nz ** 2))
        trans_mat = np.dot(pc_roty(-a), pc_rotx(b))  # 计算使得地面法向量垂直尽头的旋转矩阵（注意：不能引入绕z轴转动）

        ground_mean = bg_dep[ground_mask].mean()
        ground_dist = ground_mean if GROUND_DIST_MEAS else GROUND_DIST

        trans_mat = np.dot(pc_mov(0, 0, -ground_dist), trans_mat)  # 加入以地面中心旋转平移
        trans_mat = np.dot(trans_mat, pc_mov(0, 0, ground_dist))

        # 验证地面倾角被矫正
        pc_new = pc_trans(trans_mat, pc)
        print(fit_plane_param(pc_new))  # 前三个值应该接近0/0/1或者0/0/-1

        # 背景倾角矫正
        bg_dep, mask = img_dep_trans_remap(bg_dep, trans_mat)

        # 补洞
        bg_dep, mask = img_fill_hole(bg_dep, mask, win=3, th=4)
        bg_dep[~mask] = ground_dist
    else:
        clip_mask = np.bitwise_and(bg_dep > (GROUND_DIST - 0.2), bg_dep < (GROUND_DIST + 0.2))

        ground_mean = bg_dep[np.bitwise_and(GROUND_MASK, clip_mask)].mean()
        ground_dist = ground_mean if GROUND_DIST_MEAS else GROUND_DIST
        trans_mat = None

    ###### DEBUG #######
    # IPython.embed()
    ###### DEBUG #######
    if BACKGROUND_PLOT:
        img_show = np.flipud(bg_dep.copy())
        img_show[img_show > 3] = 3
        # plt.figure(1)
        plt_mesh(img_show)

        plt.figure(2)
        plt.imshow(bg_dep)
        plt.title('background depth')

        plt.figure(3)
        plt.imshow(bg_amp)
        plt.title('background IR')
        plt.show()

    return bg_dep, bg_amp, ground_mean, ground_dist, trans_mat


## 仅仅保留最大面积的盒子
def keep_max_area_rect(rect):
    ###### DEBUG #######
    # IPython.embed()
    ###### DEBUG #######

    idx = np.argmax([r[5] for r in rect])
    return [rect[idx]]


def edge_rect_removal(n0, n1, n2, n3, w_min=20, w_max=300, h_min=10, h_max=230):  # 靠近图像边界的矩形不检测
    vertices = np.concatenate([n0, n1, n2, n3], axis=0).reshape([-1, 2])
    v_w = np.sort(vertices[:, 0])
    v_h = np.sort(vertices[:, 1])
    return v_h[0] > h_min and v_h[-1] < h_max and v_w[0] > w_min and v_w[-1] < w_max


## 测量值滤波（注意，仅仅对单个盒子进行了测量平均）
class meas_filter_c:
    def __init__(self, sz=20):
        self.win_wid = [0 for n in range(sz)]
        self.win_len = [0 for n in range(sz)]
        self.win_hgt = [0 for n in range(sz)]
        self.sz = sz
        self.one_passed = False
        self.detect_count = 0
        self.reset_count = 0
        self.ave_w_l_h = np.array([0, 0, 0])
        return

    def reset(self, wid, len, hgt):
        self.win_wid = [wid for n in range(self.sz)]
        self.win_len = [len for n in range(self.sz)]
        self.win_hgt = [hgt for n in range(self.sz)]

    def filter(self, box_wid, box_len, box_hgt, THRESHOLD=0.05):
        if abs(box_hgt - self.ave_w_l_h[2]) > THRESHOLD or abs(box_len - self.ave_w_l_h[1]) > THRESHOLD or abs(
                        box_wid - self.ave_w_l_h[0]) > THRESHOLD:
            self.reset_count += 1
            if self.reset_count > 1:
                self.reset_count = 0
                self.detect_count = 0

        else:
            self.reset_count = 0
        self.win_wid.append(box_wid)
        self.win_len.append(box_len)
        self.win_hgt.append(box_hgt)
        self.detect_count += 1
        self.detect_count = min(self.detect_count, 20)
        self.ave_w_l_h = [np.mean(self.win_wid[-self.detect_count:]), np.mean(self.win_len[-self.detect_count:]),
                          np.mean(self.win_hgt[-self.detect_count:])]

        self.win_wid.pop(0)
        self.win_len.pop(0)
        self.win_hgt.pop(0)

        # box_wid_filter=(np.sum(self.win_wid)-np.min(self.win_wid)-np.max(self.win_wid))/(self.sz-2)
        # box_len_filter=(np.sum(self.win_len)-np.min(self.win_len)-np.max(self.win_len))/(self.sz-2)
        # box_hgt_filter=(np.sum(self.win_hgt)-np.min(self.win_hgt)-np.max(self.win_hgt))/(self.sz-2)

        return self.ave_w_l_h[0], self.ave_w_l_h[1], self.ave_w_l_h[2]


## 手动校准
def meas_corr(box_wid, box_len, box_hgt):
    K = 0.155
    S = 0.609
    H = 1.0
    return (box_wid + K) * S, (box_len + K) * S, box_hgt * H


def convert_to_binary(original_image):
    gray_image = original_image.clip(0, 255).astype(np.uint8).reshape(
        (240, 320))  # cv2.cvtColor(np.tile(np.expand_dims(original_image,-1),[1,1,3]), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)

    thresh_binary = cv2.bitwise_not(thresh)
    return thresh_binary


####################
# 入口代码
####################
#
SKIP_LIST = None  # { 180:880-180 }
PLAYBACK_END =4500 #None#480#1200 #
import pickle

if __name__ == '__main__':
    print_config()

    # GUI显示器
    viewer=cv_viewer_c(pan_hgt=IMG_HGT, pan_wid=IMG_WID, pan_num=(2,2))

    # 相机
    if PLAYBACK:
        cam = playback_cam_c(fname_dep=PLAYBACK_DEP, fname_amp=PLAYBACK_AMP, \
                             skip=PLAYBACK_SKIP, rewind=PLAYBACK_REWIND, \
                             skip_list=SKIP_LIST)  # {150:50, 300:50})
    else:
        cam = dmcam_dev_c()
        cam.set_intg(DMCAM_INTG)
        cam.set_framerate(DMCAM_FRAMERATE)
        cam.set_freq(DMCAM_FREQ)
        cam.set_pix_calib(DMCAM_PIX_CALIB)
        cam.set_hdr(DMCAM_HDR)
        #cam.set_gauss_filter(DMCAM_GAUSS)
        cam.set_median_filter(DMCAM_MED)
        if DMCAM_MODE=='2DCS':
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

    # 背景检测
    bg_dep, bg_amp, ground_mean, ground_dist, trans_mat = background_det(cam, viewer)

    # 测量主循环
    frame_cnt = 0

    flag_passed = False
    pass_count = 0
    ave_info = None
    test_id,box_id=1,1
    num_test=[1,2,3,4] #做完第几次实验换下个盒子
    num_box=len(num_test)
    test_frame=400
    LOG_FNAME = './calib_data/test_' + str(test_id) + ' _box_' + str(box_id) + '.txt'
    fp_log = open(LOG_FNAME, 'w')
    log_out=('校准将在二十秒后开始,请做好准备！\n Calibration will start in 20s, be ready! test frame:{}'.format(test_frame))
    print(log_out)
    time.sleep(20)
    while True:
        # 获取图片
        img_dep, img_amp, frame_cnt_new = cam.get_dep_amp()
        frame_cnt_new-=(BACKGROUND_SKIP+BACKGROUND_CUM)
        if frame_cnt == frame_cnt_new:
            time.sleep(0.001)
            continue
        else:
            frame_cnt = frame_cnt_new
            if frame_cnt % 10 == 0: print('[%d]' % frame_cnt)

        if frame_cnt_new%test_frame==0:
            test_id+=1
            LOG_FNAME = './calib_data/test_' + str(test_id) +' _box_' + str(box_id) + '.txt'
            if frame_cnt_new//(test_frame)==num_test[box_id-1]:
                box_id+=1
                if box_id>num_box:
                    print('结束校准\n calibration finished!')
                    exit()
                test_id=1
                LOG_FNAME = './calib_data/test_' + str(test_id) + ' _box_' + str(box_id) + '.txt'
                log_out =('请准备校准下一个盒子！\n Be ready to test with another box. (30s)')
                print(log_out)
                time.sleep(30)
            else:
                log_out =('请对同一个盒子再校准一次！\n Be ready to have another test with the same box.(10s)')
                print(log_out)
                time.sleep(10)
            fp_log = open(LOG_FNAME, 'w')
            time.sleep(0.01)



            # 有效区域屏蔽码
        mask = np.ones_like(img_dep, dtype=bool) if BACKGROUND_MASK is None else BACKGROUND_MASK.copy()

        # if MID_FILTER: img_dep=mid_filter.calc(img_dep) # 时域中值滤波
        # if IIR_FILTER: img_dep=iir_filter.calc(img_dep) # 时域平滑滤波

        # 镜头去畸变
        if MEAS_UNDISTORT:
            img_dep = cv2.undistort(img_dep, MEAS_DMAT, MEAS_DVEC)
            img_amp = cv2.undistort(img_amp, MEAS_DMAT, MEAS_DVEC)
            mask = np.bitwise_and(mask, img_dep > 0)

        if MEAS_DEP_TO_Z: img_dep = img_dep_to_z(img_dep, MEAS_F, MEAS_CX, MEAS_CY)  # 测距射线长度矫正
        # if GROUND_CORR: img_dep,mask=img_dep_trans_remap(img_dep,trans_mat,mask) #  地面倾角矫正

        # 显示深度图和强度图
        img_dep_view = img_to_cmap(img_dep.copy(), mask=None, vmin=VIEW_DMIN, vmax=VIEW_DMAX)
        cv2.putText(img_dep_view, '[%d]' % frame_cnt, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        viewer.update_pan_img_rgb(img_dep_view, pan_id=(0, 0))
        # if img_amp is not None: viewer.update_pan_img_gray(img_amp*VIEW_AMP_SCALE,pan_id=(0,1))

        # 图像预处理
        mask = np.bitwise_and(mask, img_dep < IMG_DEP_SAT_TH)  # 最大深度过滤

        if AMP_FILTER: mask = np.bitwise_and(mask, img_amp >= AMP_TH)  # 最小光强过滤

        mask_bg = (bg_dep - img_dep) > BG_NOISE_TH  # 背景扣除， 只保留背景之上的像素
        if AMP_FILTER: mask_bg = np.bitwise_or(bg_amp < AMP_TH, mask_bg)  # 最大光强过滤，太大可能是空洞
        mask = np.bitwise_and(mask, mask_bg)

        # 滤波处理
        # if FLY_NOISE_FILTER: mask=np.bitwise_and(mask,fly_noise_mask.calc(img_dep,th=FLY_NOISE_TH,win=FLY_NOISE_WIN,mode='td')) # 飞散点滤除
        if FILL_HOLE: img_dep, mask = img_fill_hole(img_dep, mask, win=3, th=4)  # 填补空洞
        # if MORPH_OPEN :      mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,KER_OPEN  )>0   # 形态学开运算（去处散点）
        # if MORPH_CLOSE:      mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,KER_CLOSE)>0   # 形态学闭运算边界凹陷去除

        # if LAPLACE_FILTER:   img_dep=img_noise_filter(img_dep,mask,it=1)        # 拉普拉斯滤波
        # if SPACE_MID_FILTER: img_dep=cv2.medianBlur(img_dep,SPACE_MID_FILTER)   # 空域中值滤波

        # 边沿检测
        if EDGE_DEP_BY_AMP:  # 基于强度图的边沿检测
            img_amp_diff = img_amp - bg_amp
            img_bin_diff = img_amp_diff > EDGE_DEP_AMP_TH
            #cv2.imshow('bin_diff', np.uint8(img_bin_diff) * 255)

            img_edge = canny(img_bin_diff, EDGE_DET_TH).astype(int)
        else:  # 基于深度图的边沿检测
            if GROUND_FILL: img_dep[~mask] = ground_dist  # 无效区域填入平均地面深度，防止产生过多边沿
            img_edge = canny(img_dep, EDGE_DET_TH).astype(int)

        # 伪彩色
        img_dep_cmap = img_to_cmap(img_dep, mask=mask, vmin=VIEW_DMIN, vmax=VIEW_DMAX)

        # 当边界检测足够“干净”时才启动立方体检测（不检测杂乱图像）
        go_on = img_edge.sum() < IMG_EDGE_POINT_MAX
        if not go_on:pass# print('stop detection, img_edge.sum():', img_edge.sum())

        # 矩形检测
        if go_on:
            h, theta, d = hough_line(img_edge)
            go_on = h.max() > HOUGH_LINE_CUM_MIN  # 当检出足够长的边界直线时才继续检测

        if go_on:
            # 提取直线参数
            th = max(HOUGH_LINE_CUM_MIN, HOUGH_LINE_TH_COEFF * h.max())  # 门限，用于直线参数提取
            accum, angles, dists = hough_line_peaks(h, theta, d, min_distance=9, min_angle=10, threshold=th)
            num0 = len(accum)
            go_on = len(accum) < LINE_NUM_MAX and len(accum) >= 4  # 当检出足够少的直线时才继续（不检测杂乱图像）
            if not go_on:pass# print('stop detection, len(accum):', len(accum))
        else:
            if flag_passed == False:
                pass_count += 1
                if pass_count > 4:
                    flag_passed = True
        ###### DEBUG #######
        # frame_cnt==159: IPython.embed()
        ###### DEBUG #######

        while go_on:
            # 检出所有矩形
            rect = find_rect_from_hough_points(angles, dists)
            num1 = len(rect)
            #print('%02d ' % num1, end='')
            if num1 == 0: break

            # 根据长短边尺寸过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if
                    verify_rect_by_size_ratio(n0, n1, n2, n3)]
            num2 = len(rect)
            #print('%02d ' % num2, end='')
            if num2 == 0: break

            # 根据面积过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if verify_rect_by_area(n0, n1, n2, n3)]
            num3 = len(rect)
            #print('%02d ' % num3, end='')
            if num3 == 0: break

            # 根据边沿图过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if
                    verify_rect_by_edge(1 - img_edge, n0, n1, n2, n3)]
            num4 = len(rect)
            #print('%02d ' % num4, end='')
            if num4 == 0: break

            # 太靠近图像边界的矩形过滤
            rect = [(n0, n1, n2, n3, q, r) for n0, n1, n2, n3, q, r in rect if edge_rect_removal(n0, n1, n2, n3)]

            # 根据重叠情况合并
            rect = rect_merge(rect)
            num5 = len(rect)
            #print('%02d ' % num5, end='')
            if num5 == 0: break

            # 只保留最大面积的盒子
            if RECT_KEEP_MAX: rect = keep_max_area_rect(rect)

            # 多帧跟踪(仅适用于静态滤波)
            # if RECT_TRK:
            #     for n0,n1,n2,n3,_,_ in rect:
            #         rect_trk.match_add(n0,n1,n2,n3,frame_cnt)
            #     rect_trk.clean_up(frame_cnt)
            #     rect=rect_trk.rect_seq.copy()

            # 计算盒子参数并显示
            c, text_y = 0, 15

            for i in range(len(rect)):
                flag_passed = False
                pass_count = 0
                # if RECT_TRK:    # 不处理跟踪门限之下的矩形（滤除那些“闪现”的矩形）
                #     if rect[i][5]<RECT_TRK_CNT_TH: continue

                # 绘制显示检测得到的矩形
                n0, n1, n2, n3 = rect[i][0], rect[i][1], rect[i][2], rect[i][3]
                draw_poly4(img_dep_cmap, n0, n1, n2, n3, thickness=2)
                draw_poly4(img_amp, n0, n1, n2, n3, color=(255, 255, 255), thickness=2)
                # 计算盒子顶面到镜头的距离
                box_dep, box_amp = calc_box_dep_amp(img_dep, img_amp, n0, n1, n2, n3, mask)
                box_wid, box_len = calc_box_wid_len(n0, n1, n2, n3, box_dep)  # 盒子长宽像素尺寸
                # 投射到三维空间平面
                if box_wid < box_len: box_wid, box_len = box_len, box_wid

                if MEAS_CONV:  # 是否转换成物理尺寸？
                    # if MEAS_WID_LEN_CORR==0:
                    #     box_wid*=box_dep/MEAS_F                 # 盒子长宽像素尺寸转成物理尺寸
                    #     box_len*=box_dep/MEAS_F
                    # else:
                    #     box_wid,box_len=box_wid_len_corr(box_wid,box_len,box_dep,box_amp)

                    if MEAS_HGT_CORR == 0:
                        box_hgt = ground_dist - box_dep  # 计算盒子的高度（扣除地面高度）
                    elif MEAS_HGT_CORR == 2:  # 使用修正模型计算盒子的高度
                        box_hgt = box_dep_to_hgt(box_dep, box_amp / 1000)

                    ######## 手动校准 #########
                    # FIXME!
                    # box_wid,box_len,box_hgt=meas_corr(box_wid,box_len,box_hgt)
                    frame_info = '%d: %03d*%03d*%04d(mm)' % (c, box_wid * 1000, box_len * 1000, box_hgt * 1000)
                    if MEAS_FILTER: box_wid, box_len, box_hgt = meas_filter.filter(box_wid, box_len, box_hgt)

                    ave_info = '%d: %03d*%03d*%04d(mm)' % (c, box_wid * 1000, box_len * 1000, box_hgt * 1000)
                else:
                    frame_info = '%d: %03d(pix)*%03d(pix)*%04d(mm), %04d' % (
                    c, box_wid, box_len, box_dep * 1000, box_amp)

                # # 屏蔽远离中心的盒子测量结果
                # cx,cy=calc_rect_cent(n0,n1,n2,n3)
                # if cy>=min(MEAS_SHOW_RANGE) and cy<=max(MEAS_SHOW_RANGE):
                # if True:
                #     err_len=box_len*1000-BOX_GROUND_TRUTH[0]

                # if flag_passed==True and box_hgt!=0:
                #     cv2.putText(img_dep_cmap, ave_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                # else:
                cv2.putText(img_dep_cmap, frame_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                text_y += 15
                c += 1
                #print('box_amp:', box_amp)

                if MEAS_CONV:
                    fp_log.write('%d,%f,%f,%f\n' % (frame_cnt, box_wid, box_len, box_hgt))
                else:
                    fp_log.write('%d,%f,%f,%f,%f\n' % (frame_cnt, box_wid, box_len, box_dep, box_amp))

            #print('')
            break

        if flag_passed == True and ave_info != None:
            cv2.putText(img_dep_cmap, ave_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # else:
        #     cv2.putText(img_dep_cmap, frame_info, (2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # 显示图像
        viewer.update_pan_img_rgb(img_dep_cmap, pan_id=(1, 0))
        edge_img=img_to_cmap(1 - img_edge, mask=None, vmin=0, vmax=1)
        viewer.update_pan_img_rgb(edge_img, pan_id=(1, 1))
        if img_amp is not None: viewer.update_pan_img_gray(img_amp * VIEW_AMP_SCALE, pan_id=(0, 1))
        # 屏幕刷新
        viewer.update()
        evt, param = viewer.poll_evt()
        if evt is not None:
            if evt == 'quit': break

        if PLAYBACK_END is not None:
            if frame_cnt == PLAYBACK_END: break

    fp_log.close()

