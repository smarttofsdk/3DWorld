##  功能描述
#   完整的RVL压缩代码，其中包含VLE编解码和数据的处理代码

import numpy as np
# 声明全局变量
nibblesWritten = 0
pBuffer = []
cnt_enc = 0
cnt_dec = 0
word = 0
nibble = 0
# VLE编码函数
def EncodeVLE(value):
    # 在Python中调用全局变量需要在函数中声明
    global word
    global nibblesWritten
    global pBuffer
    global cnt_enc
    global nibble
    while True:
        nibble = value & 0x7  # 取最低三位作为nibble
        value = value >> 3  # 输入值左移三位
        if value != 0:  # 当左移三位后不为空时，第四位保存为1
            nibble = nibble | 0x8
        word = word << 4
        word = word | nibble  # 与之前值的合并
        nibblesWritten = nibblesWritten+1  # 计数变量加一
        if nibblesWritten == 8 :
            pBuffer.append(word)  # 存满4*8=32位后，存入缓存区pBuffer
            nibblesWritten = 0
            word = 0
            cnt_enc = cnt_enc +2
        if value == 0:  # 若左移三位后为空，保存为0，该像素点的值编码结束
            break

def compressRVL(input):
    cnt = 0
    previous = 0
    size = input.shape[0]
    global nibblesWritten
    global cnt_enc
    global word
    while(cnt< size):
        zeros = 0
        nonzeros = 0
        # 统计连续零的个数
        for i in range(size-cnt):
            if input[cnt] == 0:
                zeros = zeros+1
                cnt = cnt+1
            else:
                break
        EncodeVLE(zeros) # 编码当前连续零的数值
        # 统计连续非零个数
        for j in range(size-cnt):
            if input[cnt] !=0:
                nonzeros = nonzeros+1
                cnt = cnt+1
            else:
                break
        EncodeVLE(nonzeros) # 编码当前连续非零的数值
        # 编码当前非零序列中的每个数
        for k in range(nonzeros):
            current = input[cnt-nonzeros+k]
            delta = current - previous
            positive = (delta<<1) ^ (delta>>31)
            EncodeVLE(positive)
            previous = current
    # 编码结束后，不满32bit也要补零存储
    if nibblesWritten !=0:
        pBuffer.append(word << (4*(8-nibblesWritten)))
        print(word << (4*(8-nibblesWritten)))

def DecodeVLE():
    value = 0
    bits = 28
    global nibblesWritten
    global cnt_dec
    global word
    global nibble
    while True:
        if (nibblesWritten == 0):  
            # 记录处理了多少个4bit，4*8=32个bit都被处理后，读入新的变量
            word = pBuffer[cnt_dec]
            print(word)
            cnt_dec = cnt_dec + 1
            nibblesWritten = 8

        nibble = word & 0xf0000000  # 取最高4位解码
        nibbleouthigh = nibble & 0x70000000  
        value =value |(nibbleouthigh>> (bits)) 
        word <<= 4
        nibblesWritten = nibblesWritten -1
        bits = bits - 3
        if ((nibble & 0x80000000) == 0):  
            # 当最高一位是零时，当前的值解码结束
            break
    return value,cnt_dec

def DecompressRVL(input,numpixels):
    cnt = 0
    output = np.zeros(numpixels)
    previous = 0
    global nibblesWritten
    nibblesWritten = 0
    while (cnt<numpixels):
	# 解码得到的第一个数为连续零的个数
        zeros,temp = DecodeVLE()
        # 输出zeros个零
        for i in range(zeros):
            output[cnt+i] = 0

        cnt = cnt + zeros
        # 得到连续零的个数后，下一个数表示连续不是零的数的个数
        nonzeros, temp = DecodeVLE()
        # 解码不是零的nonzeros个数
        for j in range(nonzeros):
            positive, temp = DecodeVLE() # 解码当前值
            delta = (positive >> 1) ^ -(positive & 1);
            current = previous + delta
            output[cnt+j] = current
            previous = current

        cnt = cnt + nonzeros
    print(output)

if __name__ == '__main__':
    testarray = np.array([1010, 1050, 3045, 2052, 2456, 0, 0, 0, 20, 23, 24, 34, 0, 0, 25, 38, 29])
    compressRVL(testarray)
    print(pBuffer)
    DecompressRVL(pBuffer,testarray.size)
