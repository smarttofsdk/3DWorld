##  功能描述
#   读写BIN文件
#   输入参数：
#   readfile：读入BIN文件的路径和文件名
#   savefile：写入BIN文件的路径和文件名
#   MAX_CNT：读入图片的总帧数
#   IMG_SIZE：图片的大小

readfile = open("./data/readtest.bin", 'rb') # 'rb'读入模式
savefile = open("./data/savetest.bin", 'wb') # 'wb'写入模式
IMG_HGT = 240
IMG_WID = 320
IMG_SIZE = IMG_HGT*IMG_WID
import numpy as np
for i in range(MAX_CNT):
# 此时读入的frame_test是一个一维数组，深度图每行的首尾连接
    frame_test = np.fromfile(readfile, dtype=np.uint16, count=IMG_ SIZE)
    img_dep= np.reshape(frame_test, (IMG_HGT, IMG_WID))
    savefile.write(img_dep)
savefile.close()
