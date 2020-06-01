import cv2

##  功能描述：
##  调用OpenCV函数对深度图进行中值滤波
#   输入参数：
#   img_dep：待处理深度图
#   win：滑动窗尺寸
#   输出参数：
#   img_blur：滤波结果
def depth_medianBlur(img_dep, win=3):
	img_blur=cv2.medianBlur(img_dep, win)  
	return img_blur

img_dep = cv2.imread('example_1.png', -1)
img_dep = depth_medianBlur(img_dep=img_dep, win=3)
cv.imshow("img_dep", img_dep)


