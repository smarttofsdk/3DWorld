# -*- coding: utf-8 -*-

from HPEmodel import *
from sklearn import cluster
from sklearn.externals import joblib

namelis = ['pic7_1_tof']
rootpath =  r'data/'
hpe = HPE('model/RFtof_1.pkl')

for name in namelis:

    folders = FolderImage(name, rootpath, depSuf='/')
    
    while folders.nextImage():

        img_dep = folders.get_depImage()
        dep0_c3 = cv2.convertScaleAbs(img_dep, None, 1 / 16)
        dep0_c3 = cv2.merge([dep0_c3] * 3)
        depnb_c3 = dep0_c3.copy()

 
        img_hand = hand_cut(img_dep=img_dep,
                            dmax=5000, dmin=200,  # 深度图
                            )
                            
        img_dep[img_hand == 0] = 0
        depnb_c3[img_hand == 0] = 0

        if np.sum(img_dep>0)>500:
            joints = hpe.predict(img_dep,8)
            depnb_c3 = connect_joints(depnb_c3, joints)


        cv2.imshow("dep0", dep0_c3)
        cv2.imshow('nobackground', depnb_c3)

        key = cv2.waitKey(20)
        if key == 27:
            exit(0)
        elif key == ord('q'):
            break