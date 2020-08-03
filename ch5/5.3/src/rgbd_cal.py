import numpy as np
import cv2
from ctypes import *
#from kinect_cam_new import *
from ctypes import cdll
from depth_cam_tools import *

# rgb内参
# K_rgb=np.matrix([[1044.7786, 0, 985.9435],
#         [0, 1047.2506, 522.7765],
#         [0, 0, 1]])
#
# # ir内参
# K_ir=np.matrix([[368.8057, 0, 255.5000],
#         [0, 369.5268, 211.5000],
#         [0, 0, 1]])
#
# R_ir2rgb=np.matrix([[0.9996, 0.0023, -0.0269],
#         [-0.0018, 0.9998, 0.0162],
#         [0.0269, -0.0162, 0.9995]])
#
# T_temp=np.matrix([-75.9080,
#         14.1045,
#         33.9045])

T = K_rgb*T_temp.T
R = K_rgb*R_ir2rgb*K_ir.I
# 单点转换函数
# d 原始深度/1000


# 深度图处理工具
dep_trans=depth_cam_trans_c()
dep_trans.cam_param_k(1.0/3.727806701424671e2)

# 640*480
K_rgb=np.matrix([[364.0706, 0, 311.1238],
        [0, 486.1341, 235.7895],
        [0, 0, 1]])
# 640*480
T_temp=np.matrix([-60.1376,
                  -0.8842,
                  19.1468])
# 640*480
K_ir=np.matrix([[360.8538, 0, 241.8416],
        [0, 361.1244, 203.6490],
        [0, 0, 1]])
# 640*480
R_ir2rgb=np.matrix([[1.0000, 0.0016, -0.0043],
        [-0.0015, 0.9999, 0.0113],
        [0.0043, -0.0114, 1.0006]])
T = K_rgb*T_temp.T
R = K_rgb*R_ir2rgb*K_ir.I
# # 1920*1080
# K_rgb=np.matrix([[1178.6481, 0, 959.5858],
#         [0, 1078.3064, 518.0587],
#         [0, 0, 1]])
# # 1920*1080
# K_ir=np.matrix([[407.8104, 0, 248.0241],
#         [0, 407.3778, 201.4311],
#         [0, 0, 1]])
# #1920*1080
# R_ir2rgb=np.matrix([[0.9999, 0.0028, 0.0110],
#         [-0.0026, 1.0000, -0.0041],
#         [-0.0109, 0.0040, 1.0000]])
# # 1920*1080
# T_temp=np.matrix([-69.3395,
#                   14.8735,
#                   1.1352])



# 单点转换函数
# d 原始深度/1000

def rgbd_calibration_pc(pc):
    data_len=len(pc)
    #u,v,d = dep_trans.pcloudone_to_depth(pc) # u 512 v 424
    dep_img = dep_trans.pcloudone_to_depth(pc)
    uv_depth = np.matrix([u,v,1])

    if d != 0 and d != 65535:
        uv_color = d * R * uv_depth.T + T / 1000
        RGB_X=(uv_color[0]/uv_color[2]).astype(np.uint16)
        RGB_Y=(uv_color[1]/uv_color[2]).astype(np.uint16)
    return RGB_X,RGB_Y

def rgbd_calibration(x,y,d):
    uv_depth = np.matrix([x,y,1])
    # print(uv_depth)
    X, Y = 0, 0
    if d != 0 and d != 65535:
        uv_color = d * R * uv_depth.T + T / 1000
        X=(uv_color[0]/uv_color[2]).astype(np.uint16)
        Y=(uv_color[1]/uv_color[2]).astype(np.uint16)
        # X = (X * 512 / 1920).astype(np.uint16)
        # Y = (Y * 424 / 1080).astype(np.uint16)
    return X,Y

# 图片转换函数
def depth2rgb(img_dep,img_rgb):
    img_hgt,img_wid = img_dep.shape[0],img_dep.shape[1]
    img_result = np.zeros([img_hgt,img_wid,3],dtype=np.uint8)
    for i in range(img_hgt):
        for j in range(img_wid):
            X,Y=rgbd_calibration(j,i,img_dep[i,j])
            X = int(X * img_rgb.shape[1] / 1920)
            Y = int(Y * img_rgb.shape[0] / 1080)
            if ((X >= 0 and X < img_rgb.shape[1]) and (Y >= 0 and Y < img_rgb.shape[0])):
                img_result[i,j,:]=img_rgb[Y,X,:]
    return img_result
# rgbd_calibration(112,324,0.5)

def main():
    # img_dep = cv2.imread()
    fp_dep = open('../data/depth12102.bin', 'rb')
    fp_clr = open('../data/rgb12102.bin', 'rb')
    # dll = windll.LoadLibrary('rgbd_calib_dll_64_3.dll')
    #dll = windll.LoadLibrary('wjjj_80_0.dll')
    #import os
    # dll=cdll.LoadLibrary('/home/jokes/my_file/3D_project/3D-scnaer/3D-scaner/3rd_library/rgbd_calib_dll/build/librgbd_calib_dll.so')
    # dll.display('Hello,I am linuxany.com')
    FRAME_DEP_SZ = 424*512
    FRAME_CLR_SZ = 1080*1920*3
    # FRAME_CLR_SZ = 1024 * 576 * 4
    frame_cnt= 0
    while True:
        # 从数据中读取一帧深度图
        frame_dep = np.frombuffer(fp_dep.read(2*FRAME_DEP_SZ),dtype=np.uint16)
        # frame_cnt += 1
        img_dep = frame_dep.copy().astype(np.float32).reshape([424,512]) / 1000
        img_dep_show = img_dep.copy().astype(np.uint8)
        # 从数据中读取一副彩色图
        frame_clr = np.frombuffer(fp_clr.read(FRAME_CLR_SZ), dtype=np.uint8)
        img_clr = frame_clr.reshape([1080, 1920, 3]).astype(np.uint8)
        # cv2.imshow("dep", img_dep)
        # cv2.imshow("img_rgb", img_clr)
        # img_clr = cv2.resize(img_clr, (512, 424), interpolation=cv2.INTER_NEAREST)
        # img_result=depth2rgb(img_dep,img_clr)
        frame_cnt+=1
        img_result = np.zeros((424,512,3),dtype=np.uint8)
        d_rows,d_cols=img_dep.shape[0],img_dep.shape[1]
        rows,cols = img_clr.shape[0],img_clr.shape[1]
        dll.calib(d_rows, d_cols, img_dep.ctypes.data_as(POINTER(c_float)), img_result.ctypes.data_as(POINTER(c_int)),
                  rows, cols, img_clr.ctypes.data_as(POINTER(c_int)))
        cv2.imshow("img_rgb",img_clr)
        cv2.imshow("dep",img_dep)
        cv2.imshow("result",img_result)
        cv2.waitKey(10)

    fp_dep.close()
    # fp_clr.close()

if __name__ == '__main__':
    main()