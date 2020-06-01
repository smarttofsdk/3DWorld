import cv2

##  功能描述：
##  调用OpenCV函数对深度图进行高斯滤波
#   输入参数：
#   img_dep：待处理的深度图
#   win：窗口尺寸
#   sigma：标准差
#   输出参数：
#   img_blur：滤波结果
def depth_GaussianBlur(img_dep, win=3, sigma=0.1):
	img_blur=cv2.GaussianBlur (img_dep,(win,win), sigma)
	return img_blur

img_dep = cv2.imread('example_1.png', -1)
img_dep = depth_GaussianBlur(img_dep=img_dep, win=3, sigma=0.1)
cv2.imshow("img_dep", img_dep)
cv2.waitKey(0)

