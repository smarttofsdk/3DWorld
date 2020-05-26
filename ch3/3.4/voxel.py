## 本脚本实现体素滤波下采样
## Author：Weihang Wang

import numpy as np
import cv2
import matplotlib.pyplot as plt


def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 计算边界点
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)  # 计算x y z 三个维度的最值
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)

    # 计算 voxel grid维度
    Dx = (x_max - x_min) // leaf_size + 1
    Dy = (y_max - y_min) // leaf_size + 1
    Dz = (z_max - z_min) // leaf_size + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))

    # 计算每个点的voxel索引
    h = list()  # h 为保存索引的列表
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min) // leaf_size
        hy = (point_cloud[i][1] - y_min) // leaf_size
        hz = (point_cloud[i][2] - z_min) // leaf_size
        # 当前点属于第几个像素块
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.array(h)

    # 筛选点
    h_indice = np.argsort(h)  # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    begin = 0
    for i in range(len(h_sorted) - 1):
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            # 体素块中心均值点
            point_idx = h_indice[begin: i + 1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
            begin = i

    # 返回下采样后点云
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def gen_tab(cx,cy,fx,fy,img_hgt,img_wid):
    u = (np.arange(320) - cx) / fx
    v = (np.arange(240) - cy) / fy
    tab_x = np.tile(u, img_hgt)
    tab_y = np.repeat(v, img_wid)
    return tab_x,tab_y

def depth_to_pcloud(img_dep, tab_x, tab_y):
    pc = np.zeros((np.size(img_dep), 3))
    pc[:, 0] = img_dep.flatten() * tab_x
    pc[:, 1] = img_dep.flatten() * tab_y
    pc[:, 2] = img_dep.flatten()
    return pc


# 读取深度图像
img_dep = cv2.imread('box.png', 0)
# 深度图转点云
tab_x,tab_y = gen_tab(160,120,180,180,240,320)
pointcloud = depth_to_pcloud(img_dep, tab_x, tab_y)
# 体素滤波下采样
filtered_cloud = voxel_filter(pointcloud, 5)


### 可视化结果
## plt 显示原始点云
m4 = np.array(pointcloud)
# 列表解析x,y,z的坐标
x = [k[0] for k in m4]
y = [k[1] for k in m4]
z = [k[2] for k in m4]
# 开始绘图
fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection='3d')
# 标题
plt.title('Origin PCL')
# 利用xyz的值，生成每个点的相应坐标（x,y,z）
ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
# ax.axis('scaled')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 显示
plt.show()


## plt 显示体素下采样结果
m4 = np.array(filtered_cloud)
# 列表解析x,y,z的坐标
x = [k[0] for k in m4]
y = [k[1] for k in m4]
z = [k[2] for k in m4]
# 开始绘图
fig = plt.figure(dpi=120)
ax = fig.add_subplot(111, projection='3d')
# 标题
plt.title('Voxel PCL')
# 利用xyz的值，生成每个点的相应坐标（x,y,z）
ax.scatter(x, y, z, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
# ax.axis('scaled')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 显示
plt.show()



