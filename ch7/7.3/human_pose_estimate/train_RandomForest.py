# -*- coding: utf-8 -*-
from clor import *
from sklearn.externals import joblib

def get_trainXY(img_clf, img_dep):
    """
    ## 从标签图像中转换得到训练点集

    :param img_clf: 标签图像 uint8
    :param img_dep: 深度图像 uint16
    :return: 采样点集X 标签y
    """
    mask = (img_clf > 0)
    y.extend(img_clf[mask][::sli])

    indy, indx = np.where(mask)
    indy = indy[::sli]
    indx = indx[::sli]
    IMG_HGT, IMG_WID = img_dep.shape

    inx = []
    iny = []
    for i in range(len(ux)):
        inx += [(indx + np.int32(np.round(ux[i] / img_dep[indy, indx])))]
        iny += [(indy + np.int32(np.round(uy[i] / img_dep[indy, indx])))]
    inx = np.array(inx)
    iny = np.array(iny)
    inmask = (inx > IMG_WID - 1) | (inx < 0) | (iny > IMG_HGT - 1) | (iny < 0)

    inx = inx.clip(0, IMG_WID - 1)
    iny = iny.clip(0, IMG_HGT - 1)

    ext = img_dep[iny, inx].copy()
    ext[inmask] = 0
    X.extend(ext.T)



## 文件夹中获取训练图片--tof分辨率且已去背景
def train_tof():
    global img_num
    ################################
    namelis = ['pic7_1_tof'] 
    rootpath = r'data/'
    skeletonfilename = rootpath+'{0}.skeleton'

    for name in namelis:
        folders = FolderImage(name, rootpath, skeletonfilename, 'j', depSuf='/')
        # ----------------------------------
        while folders.nextImage():
            img_num += 1
            img_dep = folders.get_depImage()
            skeletons = folders.get_skeletons()
            img_clf = label_joints_area(img_dep, skeletons, wid=5, mode='ki2tof')
            get_trainXY(img_clf, img_dep)
            


########################################################

sli = None
y = []
X = []
img_num = 0


print('loading data...')
# 获取训练数据
train_tof()
X = np.array(X)


print('Total image number:', img_num)
print('training...')

# --- 训练 --- #
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

print('train accurate: ',clf.score(X, y))

# 保存模型
joblib.dump(clf, 'model/RFtof_1.pkl')

print('done')
