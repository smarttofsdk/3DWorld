##  功能描述：
##  调用OpenCV函数对深度图进行高斯滤波
#   输入参数：
#   img_dep：待处理的深度图
#   win：窗口尺寸
#   sigma：标准差
#   输出参数：
#   img_blur：滤波结果

img_blur=cv2.GaussianBlur (img_dep,-1,(win,win), sigma)  

