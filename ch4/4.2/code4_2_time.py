##  功能描述：
##  时域中值滤波器
#   输入参数：
#   待处理的深度图
#   输出参数：
#   滤波结果
import cv2

def calc(img):
	# 读取图像尺寸
img_dep = cv2.imread(img)
    size = img.shape
	# 初始化参数
	buf_dep=np.zeros(size)
    idx=0
# 滤波
buf_dep[:,:,idx]=img_dep
img_sum=np.sum(buf_dep,axis=2)
img_max=np.max(buf_dep,axis=2)
img_min=np.min(buf_dep,axis=2)  

     return img_sum-img_max-img_min 
