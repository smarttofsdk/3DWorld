##  功能描述：
#   利用pcl库读取pcd文件
#   输入参数：
#   FNAME: pcd文件路径和名称
#   输出参数：
#   pc: 点云数组

import pcl
FNAME = './data/test.pcd' # pcd文件在计算机中存储的路径与名称
pc = pcl.load(FNAME + '.pcd') # 读取pcd文件
