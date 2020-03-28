##  功能描述
#   完整16bit深度图与3*8bit的YUV数据的相互转换

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def depthtoyuv(depth):
    w = 65535   # 16bit深度图取2的16次方
    # 对应公式（4.4.3）
    L = (depth+0.5)/w 
    n_p = 2048
    p = n_p/w

    # 对应公式（4.4.4）
    Ha = ((2*L)/p)%2.0
    Ha_mask = Ha.copy()
    Ha_mask[Ha_mask>1] = 1  # 计算Ha的mask，>1的像素位置保留
    Ha_mask[Ha_mask <= 1] = 0  # 其他像素位置（2-原始值）
    Ha[Ha>1] = 0
    Ha_1 = 2- (((2*L)/p)%2.0)
    Ha = Ha + Ha_1*Ha_mask
	
    # 对应公式（4.4.5）
    Hb = (((2 * L) - (p / 2))/ p)% 2.0
    Hb_mask = Hb.copy()
    Hb[Hb > 1] = 0  # 与Ha逻辑相同
    Hb_mask[Hb_mask <= 1] = 0
    Hb_mask[Hb_mask > 1] = 1
    Hb_1 = 2 - ((((2 * L) - (p / 2)) / p)% 2.0)
    Hb = Hb + Hb_1* Hb_mask

    return L,Ha,Hb

def yuvtodepth(L,Ha,Hb):
    w = 65535
    n_p = 2048
    p = n_p / w
    tmp = 4*(L/p)-0.5
    mL = (np.floor(tmp))%4

    # 对应公式（4.4.8）
    L_0 = L-((L-(p/8))%p)+((p/4)*mL)-(p/8)

    # 对应公式（4.4.9），并给出公式（4.4.7）中的判断结果，对应输入矩阵的mask
    delta = np.zeros_like(mL)
    mL_0 = mL.copy()
    mL_1 = mL.copy()
    mL_2 = mL.copy()
    mL_3 = mL.copy()

    mL_0[mL_0 != 0] = -1
    mL_0[mL_0==0]=1
    mL_0[mL_0==-1] = 0

    mL_1[mL_1 != 1] = -1
    mL_1[mL_1 == 1] = 1
    mL_1[mL_1 == -1] = 0

    mL_2[mL_2 != 2] = -1
    mL_2[mL_2 == 2] = 1
    mL_2[mL_2 == -1] = 0

    mL_3[mL_3 != 3] = -1
    mL_3[mL_3 == 3] = 1
    mL_3[mL_3 == -1] = 0

    # 对应公式（4.4.6）
    delta =  (p/2)*Ha*mL_0+(p / 2) * Hb*mL_1+\
    (p / 2) * (1 - Ha)*mL_2+(p / 2) * (1 - Hb)*mL_3
    depth = w*(L_0+delta)

return depth


if __name__ == '__main__':
	# 测试数据为0到2的16次方（65535）
    test=np.zeros(65535)
    for i in range(65535):
        test[i] = i

    # 编码
    L,Ha,Hb = depthtoyuv(test)

    # plot画出结果图
    leg1, = plt.plot(test,L,linewidth = 0.5,color = 'blue')
    leg2, = plt.plot(test,Ha,linewidth = 0.5,color = 'green')
    leg3, = plt.plot(test,Hb,linewidth = 0.5,color = 'red')
    legend = plt.legend([leg1,leg2,leg3],['L','Ha','Hb'])
    plt.show()

    # 解码
    depth = yuvtodepth(L,Ha,Hb)
    leg1, = plt.plot(test,L,linewidth = 0.5,color = 'blue')
    leg2, = plt.plot(test,Ha,linewidth = 0.5,color = 'green')
    leg3, = plt.plot(test,Hb,linewidth = 0.5,color = 'red')
    legend = plt.legend([leg1,leg2,leg3],['L','Ha','Hb'])
    plt.show()
