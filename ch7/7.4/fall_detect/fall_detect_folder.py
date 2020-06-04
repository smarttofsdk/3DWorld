# -*- coding: utf-8 -*-
"""
文件夹图片中采集数据预测

"""
from clor import *

rootpath = r'data/'
namelis = ['fallData']

denoise_kernelo = np.ones((3, 3), np.uint8)

for name in namelis:
    # joints = {}
    flag = 0
    fall_num = 0
    fine_num = 0
    isFallDown = 0
    tflag =0

    spd = CalMeanSpeed()
    folders = FolderImage(name, rootpath)
    print(name)

    # 获取背景图片
    if folders.get_bgImage(10) is not None:
        dep_bg = folders.get_bgImage(10)
    else: 
        continue


    while folders.nextImage():
        img_dep = folders.get_depImage()
        Z = folders.get_irImage()

        img_amp = cv2.convertScaleAbs(Z, None, 1)
        img_fall = cv2.convertScaleAbs(img_dep, None, 1 / 16)
        img_fall = cv2.merge([img_fall] * 3)


        #去背景
        if dep_bg is not None:
            img_dep[np.abs(img_dep.copy() - dep_bg) < 80] = 0

        img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                            amp_th=80,  # 红外图
                            dmax=5000, dmin=200,  # 深度图
                            # cutx=40, cuty=10  # 图像四周区域
                            )
        img_hand = cv2.morphologyEx(img_hand, cv2.MORPH_OPEN, denoise_kernelo, iterations=2)
        img_dep[img_hand == 0] = 0
        depnb = cv2.convertScaleAbs(img_dep, None, 1 / 16)
        depnb = cv2.merge([depnb] * 3)


        _, markers, stats, centroids = cv2.connectedComponentsWithStats(img_hand)
        area_th = 500

        for i in range(1, len(stats)):
            if stats[i][4] > area_th:  # todo *d
                center = centroids[i].astype('int')
                hum_stat = stats[i]
                img_hand[markers != i] = 0
                img_dep[img_hand == 0] = 0
                # dc = np.average(img_dep[center[1] - 3:center[1] + 4, center[0] - 3:center[0] + 4])
                # img_hand = hand_cut(img_dep=img_dep, img_amp=img_amp,
                #                     dmax=dc + 500, dmin=dc - 500,  # 深度图
                #                     )

                depnb = cv2.convertScaleAbs(img_dep, None, 1 / 16)
                depnb = cv2.merge([depnb]*3)
                cv2.rectangle(depnb, (hum_stat[0], hum_stat[1]), (hum_stat[0] + hum_stat[2], hum_stat[1] + hum_stat[3]),color=[0, 255, 0], thickness=2)

                cbk = hum_stat[3] / hum_stat[2]

                # 速度满足要求
                vy=spd.meanSpeed(center, img_dep)[1]
                if vy > 3:
                    flag = 1
                if flag == 1 and (hum_stat[3] < 120 or cbk < 1):
                    fall_num = fall_num + 1
                if flag == 1 and fall_num == 3:
                    isFallDown = 1
                    # 发生跌倒，这里加响应
                    #########################

                    # simpleaudio_play('falldown_ring.wav')

                    ##########################
                    fall_num = 0
                    flag = 0

                # 高度 角度要求
                if hum_stat[3] > 180 or cbk > 2:
                    fine_num = fine_num + 1
                if fine_num == 5:
                    flag = 0
                    fine_num = 0
                    isFallDown = 0

                break


        if isFallDown == 1:
            cv2.putText(img_fall, "FALL DOWN!", (200, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
            cv2.circle(img_fall, (250, 50), 20, (0, 0, 255), -1)
        else:
            cv2.putText(img_fall, "I'm FINE!", (200, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
            cv2.circle(img_fall, (250, 50), 20, (0, 255, 0), -1)

        cvm.imshow("img", img_fall)
        # cv2.imshow('skeleton_depth_map', img_dep0)
        cvm.imshow('nobackground', depnb)

        key = cv2.waitKey(50)
        if key == 27:
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('q'):
            break
        elif key == ord('t'):
            tflag ^= 1

        if tflag:
            folders.img_id-=1

    cv2.waitKey(5)
cv2.destroyAllWindows()