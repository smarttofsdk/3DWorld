# -*- coding: utf-8 -*-

from clor2 import *

from sklearn.externals import joblib
from sklearn.cluster import MeanShift


class HPE:
    def __init__(self, model_path):
        # 载入模型
        self.clf = joblib.load(model_path)

    def predict(self, img_dep, sli=4, proba_threshold=0.5):
        """
        # 随机森林分类 + Meanshift聚合

        :param img_dep: 深度图
        :param sli:  采样间隔
        :param proba_threshold:  信任阈值
        :return: 字典，关节点坐标位置
        """
        indy, indx = np.where(img_dep > 0)
        indy = indy[::sli]
        indx = indx[::sli]

        inx = []
        iny = []
        IMG_HGT, IMG_WID = img_dep.shape
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
        X = ext.T

        # 随机森林分类
        test_y = self.clf.predict(X)
        proba = np.max(self.clf.predict_proba(X), axis=1)

        mask = (proba > proba_threshold)
        indy = indy[mask]
        indx = indx[mask]
        test_y = test_y[mask]
        msX = np.stack((indy, indx), axis=1)

        # meanshift 聚类关节点
        joints_predict = {}
        for i in np.unique(test_y):
            mask = (test_y == i)
            # ms = KMeans(n_clusters=1)
            ms = MeanShift(bandwidth=20, bin_seeding=True)

            cluy = ms.fit(msX[mask])
            joints_predict[i] = ms.cluster_centers_[0][::-1]

        return joints_predict
