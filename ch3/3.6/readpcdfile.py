##  功能描述：
#   利用pcl库读取pcd文件
#   输入参数：
#   FNAME: pcd文件路径和名称
#   输出参数：
#   pc: 点云数组

## 请执行pip install pclpy 安装pcl python包（暂只支持windows）
import pclpy
from pclpy import pcl


# 读取3.3节保存的pcd文件“box.pcd”
# 实例化一个指定类型的点云对象，并将文件读到对象里
obj = pclpy.pcl.PointCloud.PointXYZ()
pcl.io.loadPCDFile('box.pcd', obj)
# 显示点云
viewer = pcl.visualization.PCLVisualizer('PCD viewer')
viewer.addPointCloud(obj)
while (not viewer.wasStopped()):
    viewer.spinOnce(100)
