# -*- coding: utf-8 -*-

from HPEmodel import *
from sklearn import cluster
from sklearn.externals import joblib

rootpath = r'../testData'
namelis = os.listdir(rootpath)

hpe = HPE('RF_model.pkl')
denoise_kernel = np.ones((3, 3), np.uint8)

for name in namelis:

    folders = FolderImage(name, rootpath)
    dep_bg = folders.get_bgImage(10)
    while folders.nextImage():

        img_amp = folders.get_irImage()
        img_amp = cv2.normalize(img_amp,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)

        img_dep = folders.get_depImage()
        dep0_c3 = cv2.convertScaleAbs(img_dep, None, 1 / 16)
        dep0_c3 = cv2.merge([dep0_c3] * 3)
        depnb_c3 = dep0_c3.copy()

        #去背景
        if dep_bg is not None:
            img_dep[np.abs(img_dep - dep_bg) < 500] = 0
            # img_dep[2 * np.abs(img_dep - dep_bg) / (img_dep + dep_bg + 100) < 0.2] = 0

        img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                            # amp_th=10,  # 红外图
                            dmax=5000, dmin=200,  # 深度图
                            )
        img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernel, iterations=2)
        img_dep[img_hand == 0] = 0
        depnb_c3[img_hand == 0] = 0

        if np.sum(img_dep>0)>500:
            joints = hpe.predict(img_dep,8)
            depnb_c3 = connect_joints(depnb_c3, joints, mode='diedao')

        _, markers, stats, centroids = cv2.connectedComponentsWithStats(img_hand)
        area_th = 4000
        center = None
        hum_stat = None
        skel_detect = 0
        img_hand_cp = img_hand.copy()
        img_dep_cp = img_dep.copy()


        for i in range(1, len(stats)):
            if stats[i][4] > area_th:
                center = centroids[i].astype('int')
                hum_stat = stats[i]

                img_hand = img_hand_cp.copy()
                img_dep = img_dep_cp.copy()
                img_hand[markers!=i]=0
                img_dep[img_hand == 0] = 0
                skel_detect = 1
                dc = np.average(img_dep[center[1] - 3:center[1] + 4, center[0] - 3:center[0] + 4])
                img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                                    dmax=dc + 500, dmin=dc - 500,  # 深度图
                                    )
                img_dep[img_hand == 0] = 0
                depnb_c3[img_hand == 0] = 0

                if np.sum(img_dep>0)>500:
                    joints = hpe.predict(img_dep.T,8)
                    depnb_c3 = connect_joints(depnb_c3.T, joints, mode='diedao').T
                break



        cvm.imshow("dep0", dep0_c3)
        cvm.imshow('nobackground', depnb_c3)

        key = cv2.waitKey(20)
        if key == 27:
            exit(0)
        elif key == ord('q'):
            break