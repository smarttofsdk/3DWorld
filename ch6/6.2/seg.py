#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from open3d import *
from pcl import point_cloud

# file = r"./201812_dep.bin"
# file_dep = open(file, 'rb')
# image = np.frombuffer(file_dep.read(2 * 1160 * 320 * 240), dtype=np.uint16)
# while 1:
#     image = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16)
#     image = (image/20).reshape(240,320).astype(np.uint8)
#     cv2.imwrite("box.png", image)

image = cv2.imread("box.png", 0)
# image = cv2.imread("box.png", 0)
# cv2.imshow("src", image)
# cv2.waitKey(0)
# plt.title("source image"), plt.xticks([]), plt.yticks([])
# plt.subplot(121),plt.imshow(image, "gray")
# plt.title("Image")
# plt.subplot(122),plt.hist(image.ravel(), 100, [0,255])
# plt.title("Histogram")
# plt.show()

ret1, th1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("seg", th1)
cv2.waitKey(0)
contour, h = cv2.findContours(th1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(th1, contour, -1, 0, -1)
# cv2.imshow("ctr", th1)
max_area = 0
for c in contour:
    cur_area = cv2.contourArea(c)
    if max_area < cur_area:
        max_area = cur_area
        c_max = c
for c in contour:
    if c is not c_max:
        cv2.drawContours(th1, [c], -1, 0, -1)
# img = cv2.imread("src.png", -1)
# img[img>4000] = 0
img = image * th1
gray = np.float32(img.copy())
dst = cv2.cornerHarris(gray, 2, 3, 0.03)
dst=cv2.dilate(dst,None)
idx = np.argwhere(dst >= (0.58 * dst.max()))
    # clr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # clr[dst > 0.05 * dst.max()] = [0, 0, 255]
# cv2.imshow("ctr", th1)
# cv2.imshow("bin", img)
# cv2.circle(img, (-1,80), 60, 80, -1)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
cv2.drawContours(img, [c_max], -1, (0, 255, 255), 2)
# cv2.putText(img, "Rectangle", (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 1)
# cv2.imwrite("box_hu.png", img)
# Hu = cv2.HuMoments(cv2.moments(c_max)).flatten()
# print("hu features=", Hu)
for i in idx:
    cv2.circle(image, (i[1],i[0]), 2, (255, 255, 0), -1)
cv2.imshow("clr", image)
cv2.imshow("img", img)
cv2.imwrite("conto.png", img)
cv2.imwrite("harris.png", image)
cv2.waitKey(0)

## visualize point cloud
pc = point_cloud(img).reshape((-1, 3))
pc = pc[~np.all(pc == 0, axis=1)]
# pc[:,1]-=25
# pc[:,2] -= 150
pcd = PointCloud()
pcd.points = Vector3dVector(pc)
draw_geometries([pcd])
cv2.waitKey(0)


