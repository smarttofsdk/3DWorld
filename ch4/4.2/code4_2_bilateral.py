##  功能描述：
##  调用OpenCV函数对深度图进行双边滤波
#   输入参数：
#   src：原图像
#   d：像素的邻域直径，默认为-1
#   sigmaColor：颜色空间的标准方差，一般较大
#   sigmaSpace：坐标空间的标准方差（像素单位），相对较小
#   输出参数：
#   img_blur 滤波结果
import cv2
img_blur=cv2.bilateralFilter(src=img_dep,d=-1,sigmaColor=100, sigmaSpace=15)

