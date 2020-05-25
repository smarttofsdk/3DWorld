##  功能描述
#   点云映射回深度图
#   输入参数：
#   pc: 输入点云矩阵，每个点对应一行数据坐标(x,y,z)
#   cx、cy、kx、ky：相机的内参
#   img_hgt: 深度图的高
#   img_wid: 深度图的宽
#   输出参数：
#   img_dep:转换得到的深度图
#   valid:深度图中有效像素点的个数

import numpy as np
# 防止出现除数为零的情况
ToF_CAM_EPS = 1.0e-16

def pcloud_to_depth(pc,cx,cy,fx,fy,img_hgt,img_wid):
    # 计算点云投影到传感器的像素坐标
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    kzx = z / fx
    kzy = x / fy

    u = np.round(x / (kzx + ToF_CAM_EPS) + cx).astype(int)
    v = np.round(y / (kzy + ToF_CAM_EPS) + cy).astype(int)

    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u <img_wid)), np.bitwise_and((v >= 0), (v <img_hgt)))
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    img_dep = np.full((img_hgt, img_wid), np.inf)
    for ui, vi, zi in zip(u_valid, v_valid, z_valid):
        # 近距离像素屏蔽远距离像素
        img_dep[vi, ui] = min(img_dep[vi, ui], zi)
        valid = np.bitwise_and(~np.isinf(img_dep), img_dep > 0.0)
    return img_dep,valid
