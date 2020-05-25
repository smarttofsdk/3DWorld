##  功能描述
#   深度图的伪彩色转换
#   输入参数：
#   深度图（16 bit格式）
#   输出参数：
#   伪彩色图
import cv2
import numpy as np

# 读取深度图并8bit可视化
img_src = cv2.imread("src.png", -1)
img_dep = (img_src / 16).astype(np.uint8)
cv2.imshow("img_dep", img_dep)

# 图像归一化处理
img_dep = (img_dep - np.min(img_dep)) / (np.max(img_dep) - np.min(img_dep)) * 255
img_dep = img_dep.astype(np.uint8)
# 伪彩色图转换
img_rgb = cv2.applyColorMap(img_dep, cv2.COLORMAP_JET)

cv2.imshow("img_rgb", img_rgb)
cv2.waitKey(0)
