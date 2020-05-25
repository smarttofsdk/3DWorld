##  功能描述
#   深度图转换为点云图
#   输入参数：
#   img_dep：输入深度图，数据格式为uint16
#   输出参数：
#   pointcloud：输出点云数组，是3*n的浮点数数组，每行为一个点坐标（x，y，z）

import numpy as np
import cv2
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

def depth_to_pcloud(img_dep, tab_x, tab_y):
    pc = np.zeros((np.size(img_dep), 3))
    pc[:, 0] = img_dep.flatten() * tab_x
    pc[:, 1] = img_dep.flatten() * tab_y
    pc[:, 2] = img_dep.flatten()
    return pc

if __name__ == '__main__':
    # cam_parameter = [cx,cy,fx,fy,hgt,wid]
    img_dep = cv2.imread('box.png', 0)
    cv2.imshow("box", img_dep)
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
