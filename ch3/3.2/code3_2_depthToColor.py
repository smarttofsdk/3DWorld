##  功能描述
#   深度图的伪彩色转换
#   输入参数：
#   深度图（16 bit格式）
#   输出参数：
#   伪彩色图
import cv2
# 读取深度图
img_dep = cv2.imread("box.png", -1)

# 伪彩色图转换
img_rgb = cv2.applyColorMap(img_dep, cv2.COLORMAP_RAINBOW)
