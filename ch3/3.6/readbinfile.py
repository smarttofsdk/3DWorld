##  功能描述
#   读写BIN文件
#   输入参数：
#   readfile：读入BIN文件的路径和文件名
#   savefile：写入BIN文件的路径和文件名
#   MAX_CNT：读入图片的总帧数
#   IMG_SIZE：图片的大小
####bin文件链接：https://pan.baidu.com/s/1SK_I4oVAgor-EV0QZvIPSw
####提取码：txi7

import numpy as np
import cv2

file_dep = open("20180823-2.bin", 'rb') # 'rb'读入模式

while True:
    img_d = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16)
    img_d = img_d.reshape(240, 320)

    img_d = (img_d / 20).astype(np.uint8)
    cv2.imshow("depth", img_d)
    cv2.waitKey(20)


