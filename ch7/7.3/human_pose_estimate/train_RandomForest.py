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
    print(len(indx))

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

## NTU数据集中取训练数据
def train_zip():
    global img_num
    ############################   压缩包中
    namelis = [f'S002C001P003R001A{i:03}'for i in range(1, 26)]
    # 读取zip文件中的深度map
    S = 'e' * 12 + 'f' * 6
    si = int(namelis[0][2:4])
    zp = zipfile.ZipFile('%s:/nturgbd_depth_masked_s%03d.zip' % (S[si], si))
    # ---------------------------
    for name in namelis:
        depthmapsfolder = f'nturgb+d_depth_masked/{name}/'
        fn = subdir(depthmapsfolder, zp.namelist())
        len_fn = len(fn)

        # 读取关节点文件
        skeletonfilename = f'F:/nturgbd_skeletons/nturgb+d_skeletons/{name}.skeleton'
        bodyinfo = read_skeleton_file(skeletonfilename)
        # ----------------------------------
        for num in range(0, len_fn, 2):
            img_num += 1
            img_dep = cv2.imdecode(np.array(bytearray(zp.read(fn[num]))), -1).astype(np.float32)

            img_clf = label_joints_area(img_dep, bodyinfo[num], wid=5)

            img_hand = np.zeros_like(img_dep, 'uint8')
            img_hand[img_dep > 0] = 1

            get_trainXY(img_clf, img_dep)
    zp.close()


## 文件夹中获取训练图片
def train_folder():
    global img_num
    ################################自制Kinect 图片
    # namelis = ['pic6_1', 'pic6_2', 'pic9']
    namelis = ['tof0417-%d' % i for i in range(4, 21)]
    rootpath = r'E:\TOF_DATA\fallData'
    denoise_kernelo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    skeletonfilename = 'E:/pyCode/fall_skeletons/fallData/{0}.skeleton'

    for name in namelis:
        folders = FolderImage(name, rootpath, skeletonfilename, 'diedao')
        while folders.nextImage():
            img_num += 1
            img_dep = folders.get_depImage()
            img_amp = folders.get_irImage()

            # img_dep[2 * np.abs(img_dep - dep_bg) / (img_dep + dep_bg + 100) < 0.1] = 0
            img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                                amp_th=15,  # 红外图
                                dmax=3000, dmin=1000,  # 深度图
                                cutx=60, cuty=10  # 图像四周区域
                                )
            img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernel)
            img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_CLOSE, denoise_kernel)
            img_dep[img_hand == 0] = 0

            bodys = folders.get_skeletons()
            if bodys is not None:
                img_clf = label_joints_area(img_dep, bodys, wid=10)
                get_trainXY(img_clf, img_dep)



## 文件夹中获取训练图片
def train_folder_tof():
    global img_num
    ################################自制Kinect 图片
    rootpath = r'E:\pyCode\PicData'
    namelis = os.listdir(rootpath)
    # namelis = ['tof0417-%d'%i for i in range(1,56)]  # , 'data002','data003','data103']
    skeletonfilename = 'E:/pyCode/fall_skeletons/PicData/{0}.skeleton'

    denoise_kernel = np.ones((3, 3), np.uint8)
    # denoise_kernelo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    record_dict = read_dict('record_dict.txt')

    for name in namelis:
        folders = FolderImage(name, rootpath, skeletonfilename, 'diedao')
        dep_bg = folders.get_bgImage(10)
        img_id = -1

        while folders.nextImage():
            img_id += 1
            if not record_dict.get(name).get(img_id):
                continue

            img_num += 1
            img_amp = folders.get_irImage()
            img_amp = cv2.normalize(img_amp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            img_dep = folders.get_depImage()
            frame = cv2.convertScaleAbs(img_dep, None, 1 / 16)
            frame = cv2.merge([frame, frame, frame])
            depnb_c3 = frame.copy()

            # 去背景
            if dep_bg is not None:
                img_dep[np.abs(img_dep - dep_bg) < 500] = 0
                # img_dep[2 * np.abs(img_dep - dep_bg) / (img_dep + dep_bg + 100) < 0.2] = 0

            img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                                # amp_th=10,  # 红外图
                                dmax=5000, dmin=200,  # 深度图
                                )
            img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernel, iterations=2)
            img_dep[img_hand == 0] = 0

            _, markers, stats, centroids = cv2.connectedComponentsWithStats(img_hand)
            area_th = 2000
            center = None
            hum_stat = None
            skel_detect = 0

            for i in range(1, len(stats)):
                if stats[i][4] < area_th:
                    img_hand[markers == i] = 0

            img_dep[img_hand == 0] = 0

            bodys = folders.get_skeletons()
            if bodys is not None:
                img_clf = label_joints_area(img_dep, bodys, wid=5)
                get_trainXY(img_clf, img_dep)


## 文件夹中获取训练图片--已转成tof分辨率且去背景
def train_tof():
    global img_num
    ################################自制Kinect 图片
    namelis = ['pic6_1']  # , 'pic6_2', 'pic7_2', 'pic7_1', 'pic7_3', 'pic9', 'pic8_1', 'pic8_2', 'pic8_3']
    rootpath = '../PicData/'
    skeletonfilename = '../picData/skeletons/{0}.skeleton'

    for name in namelis:
        folders = FolderImage(name, rootpath, skeletonfilename, 'j', '_tof/')
        # ----------------------------------
        while folders.nextImage():
            img_num += 1
            img_dep = folders.get_depImage()
            skeletons = folders.get_skeletons()
            img_clf = label_joints_area(img_dep, skeletons, wid=5, mode='ki2tof')
            get_trainXY(img_clf, img_dep)
        tm.timePass()  #输出运行时间


########################################################

sli = None
y = []
X = []
img_num = 0

tm= TimeRecord()

# 获取训练数据
train_folder()
X = np.array(X)

tm.timePass()
tm.totelTime()

print('Total image number:', img_num)
print('Total point num: ', X.shape)

# exit(0)

# --- 训练 --- #
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

tm.timePass()

print(clf.score(X, y))
# jg = np.zeros([26,26])
# for i in range(len(y)):
#     jg[y[i],test_y[i]]+=1

joblib.dump(clf, 'RFtof_1023_2.pkl')

print('done')
