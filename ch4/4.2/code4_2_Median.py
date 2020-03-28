##  功能描述：
##  调用OpenCV函数对深度图进行中值滤波
#   输入参数：
#   img_dep：待处理深度图
#   win：滑动窗尺寸
#   输出参数：
#   img_blur：滤波结果
import cv2
img_blur=cv2.medianBlur(img_dep, win)  

