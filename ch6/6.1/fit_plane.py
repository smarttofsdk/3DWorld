from open3d import *
import cv2
import dmcam
import numpy as np
import math

DIM = (320, 240)
K=np.array([[210.22940603469698, 0.0, 161.52480497751307], [0.0, 209.59051207712847, 121.96419404154912], [0.0, 0.0, 1.0]])
D=np.array([[-0.15851750854614136], [0.2438988143004798], [-0.6652399895369868], [0.7028902448792054]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

IMG_HGT = 240
IMG_WID = 320
cx = 157.262    # Principle point that is usually at the image of center
fx = 210.783    # Focal Length
cy = 122.083    # Principle point that is usually at the image of center
fy = 204.817    # Vertical Focal Length
hgt = 240
wid = 320

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

# 平面拟合
def fit_plane_param(pc):
    print(pc.shape)
    num, _ = pc.shape
    pc_ext=np.hstack((pc,np.ones((num,1))))
    R=np.dot(pc_ext.T,pc_ext)
    _,V=np.linalg.eigh(R.astype(np.float64))
    v0=V[:,0]
    # 法向量(长度归一化)
    nx,ny,nz,d=v0[0,],v0[1],v0[2],v0[3]
    k=1.0/np.sqrt(nx**2+ny**2+nz**2)
    nx*=k
    ny*=k
    nz*=k
    # 计算平面经过的一点
    d*=k
    px=py=pz=0
    if d!=0:
        if nx != 0: px=-d/nx
        elif ny != 0: py=-d/ny
        elif nz != 0: pz=-d/nz
    return nx,ny,nz,d,px,py,pz

# 计算距离
def calc_point_to_plane_dist(x,y,z,nx,ny,nz,px,py,pz):
    return ((x-px)*nx+(y-py)*ny+(z-pz)*nz) / math.sqrt(nx**2+ny**2+nz**2)


# cam_parameter = [cx,cy,fx,fy,hgt,wid]
img_dep = cv2.imread("src.png", -1)
# img_dep = cv2.blur(img_dep,(3,3))
# img_dep = cv2.blur(img_dep,(3,3))
# img_dep = cv2.remap(img_dep, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
img_dep[img_dep>4000] = 0
tab_x,tab_y = gen_tab(cx,cy,fx,fy,hgt,wid)
pointcloud = depth_to_pcloud(img_dep,tab_x,tab_y)
nx,ny,nz,d,px,py,pz = fit_plane_param(pointcloud)
print(nx,ny,nz,d,px,py,pz)


for m in range(76800):
    i = pointcloud[m]
    print("**********")
    x = i[0]
    y = i[1]
    z = i[2]
    dist = calc_point_to_plane_dist(x,y,z,nx,ny,nz,px,py,pz)
    for j in range(3):
        print("point=",i[j])
    print("dist = ", dist)
    # if dist < 150 or x < -500 or x > 500:
    #     for j in range(3):
    #         pointcloud[m][j] = 0
    # if z < 1200 :
    #     for j in range(3):
    #         pointcloud[m][j] = 0



pcl = pointcloud.copy()

pcd = PointCloud()
pcd.points = Vector3dVector(pcl)
draw_geometries([pcd])
