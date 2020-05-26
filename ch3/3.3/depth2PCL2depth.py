##  功能描述
#   深度图与点云图互相转换
#   输入参数：
#   img_dep：输入深度图，数据格式为uint16
#   输出参数：
#   pointcloud：输出点云数组，是3*n的浮点数数组，每行为一个点坐标（x，y，z）
#   深度图：点云转回深度图

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# 防止出现除数为零的情况
ToF_CAM_EPS = 1.0e-16
IMG_WID = 320
IMG_HGT = 240
## 计算速算表
# tab_x[u,v]=(u-u0)*fx
# tab_y[u,v]=(v-v0)*fy
# 通过速算表，计算像素位置(u,v)对应的物理坐标(x,y,z)
# x=tab_x[u,v]*z, y=tab_y[u,v]*z
# 注意：为了方便使用，tab_x和tab_y矩阵被拉直成向量存放
def gen_tab(cx,cy,fx,fy,img_hgt,img_wid):
    u = (np.arange(IMG_WID) - cx) / fx
    v = (np.arange(IMG_HGT) - cy) / fy
    tab_x = np.tile(u, img_hgt)
    tab_y = np.repeat(v, img_wid)
    return tab_x,tab_y

## 深度图转点云
def depth_to_pcloud(img_dep, tab_x, tab_y):
    pc = np.zeros((np.size(img_dep), 3))
    pc[:, 0] = img_dep.flatten() * tab_x
    pc[:, 1] = img_dep.flatten() * tab_y
    pc[:, 2] = img_dep.flatten()
    return pc

## 点云转深度图
def pcloud_to_depth(pc,cx,cy,fx,fy,img_hgt,img_wid):
    # 计算点云投影到传感器的像素坐标
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    u = (np.round(x * fx / (z + ToF_CAM_EPS) + cx)).astype(int)
    v = (np.round(y * fy / (z + ToF_CAM_EPS) + cy)).astype(int)

    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u <img_wid)), np.bitwise_and((v >= 0), (v <img_hgt)))
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    img_dep = np.zeros((IMG_HGT, IMG_WID))
    for ui, vi, zi in zip(u_valid, v_valid, z_valid):
        # 替代0像素值点
        img_dep[vi][ui] = max(img_dep[vi][ui], zi)
    return img_dep

if __name__ == '__main__':
    # 读取深度图像
    img_dep = cv2.imread('box.png', 0)
    cv2.imshow("SRC", img_dep)
    # 深度图转点云
    tab_x,tab_y = gen_tab(160,120,180,180,240,320)
    pointcloud = depth_to_pcloud(img_dep, tab_x, tab_y)
    ## plt 显示
    m4 = np.array(pointcloud)
    # 列表解析x,y,z的坐标
    x = [k[0] for k in m4]
    y = [k[1] for k in m4]
    z = [k[2] for k in m4]
    # 开始绘图
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    # 标题
    plt.title('point cloud')
    # 利用xyz的值，生成每个点的相应坐标（x,y,z）
    ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
    # ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 显示
    plt.show()
    # 保存pcd文件
    PCD_FILE_PATH = os.path.join('box.pcd')
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)
    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')
    # 得到点云点数
    point_num = pointcloud.shape[0]
    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')
    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(pointcloud[i, 0]) + ' ' + str(pointcloud[i, 1]) + ' ' + str(pointcloud[i, 2])
        handle.write(string)
    handle.close()

    ## 点云转深度图，结果与原始输入图像一致
    img_cvt = pcloud_to_depth(pointcloud, 160,120,180,180,240,320)
    img_cvt = img_cvt.astype(np.uint8)
    cv2.imshow("PCL2Depth", img_cvt)
    cv2.waitKey(0)
