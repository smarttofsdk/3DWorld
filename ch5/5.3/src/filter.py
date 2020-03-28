import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from global_cfg import *
from depth_cam_tools import *
class filter():
    @staticmethod
    def remove_neighbors(pc,min_dis,mean_k=6):
        kDTree = KDTree(pc, leaf_size=5)
        dx, idx_knn = kDTree.query(pc[:, :], k=mean_k)
        dx, idx_knn = dx[:, 1:], idx_knn[:, 1:]
        distances = np.sum(dx, axis=1) / (mean_k - 1.0)
        valid_distances = np.shape(distances)[0]

        # Estimate the mean and the standard deviation of the distance vector
        sum = np.sum(distances)
        sq_sum = np.sum(distances ** 2)

        mean = sum / float(valid_distances)
        variance = (sq_sum - sum * sum / float(valid_distances)) / (float(valid_distances) - 1)
        stddev = np.sqrt(variance)  # 标准差

        # a distance that is bigger than this signals an outlier
        if min_dis==None:
            min_dis =(mean + stddev)*0.4
        idx = np.nonzero(distances > min_dis)

        new_pc = np.copy(pc[idx])
        # print(valid_distances-new_pc.shape[0],'outliers had been removed!')
        return new_pc

    @staticmethod
    def remove_flypoint(src,threshold):#TODO

        pass

    @staticmethod
    def remove_overlap_points(pc_with_n,pc_model):#TODO needs implement

        pass

    @staticmethod
    def remove_edge_points(pc,normals,row_threshold=0.15,col_threshold=0.02):#纵向别除了，再除头秃了
        mask=np.zeros((normals.shape[0],1)).astype('bool')
        idx=0
        for normal in normals:
            len=np.linalg.norm(normal)
            if np.abs(normal[2]/(normal[0]+0.0000000000001))<row_threshold or np.abs(normal[2])<col_threshold:
                mask[idx]=False
            else:
                mask[idx]=True
            idx+=1
        normals = normals[(mask).flatten(), :]
        pc_cut = pc[(~mask).flatten(), :]
        pc = pc[(mask).flatten(), :]

        return normals,pc,pc_cut

    @staticmethod
    def remove_isolation(pc_normal_fidx,mean_k=6,std_dev=1.2):#using statistic filter,para should be modified
        if pc_normal_fidx.shape[1]==7:
            pc_normal=pc_normal_fidx[:,1:7]
        else:
            pc_normal=pc_normal_fidx
        pc=pc_normal[:,0:3]
        kDTree = KDTree(pc,leaf_size = 5)
        dx,idx_knn=kDTree.query(pc[:, :],k =mean_k)
        dx,idx_knn=dx[:,1:],idx_knn[:,1:]
        distances=np.sum(dx, axis=1)/(mean_k - 1.0)
        valid_distances = np.shape(distances)[0]

        #Estimate the mean and the standard deviation of the distance vector
        sum = np.sum(distances)
        sq_sum = np.sum(distances**2)

        mean = sum / float(valid_distances)
        variance = (sq_sum - sum * sum / float(valid_distances)) / (float(valid_distances) - 1)
        stddev = np.sqrt (variance)#标准差

        # a distance that is bigger than this signals an outlier
        distance_threshold = mean+std_dev*stddev
        idx = np.nonzero(distances < distance_threshold)

        new_pc_fidx= np.copy(pc_normal_fidx[idx])
        #print(valid_distances-new_pc.shape[0],'outliers had been removed!')
        return new_pc_fidx
        # new_color = np.copy(color[idx])

    @staticmethod
    def cube_filter(img_dep,img_dep_bg,dmin=0.2,dmax=1,distance=0.2,CUT_IMG=True):
        if CUT_IMG==True:
            #img_dep = get_rect(img_dep, 100,100,200,250)#wood man
            img_dep = filter.get_rect(img_dep, 100, 150, 200, 175)  # wood man
            # img_dep = get_rect(img_dep,50, 50, 250, 250)#new tof
            # img_dep = get_rect(img_dep, 150, 150, 150, 120)#cube
            # img_dep = get_rect(img_dep, 180, 150, 80, 100)#clock
        mask=(img_dep > dmin) * (img_dep < dmax) * (np.abs(img_dep_bg - img_dep)>distance)
        # pc = dep_trans.depth_to_pcloud(img_dep, mask)
        return mask

    @staticmethod
    def get_rect(dep_image, start_x, start_y, width, height):  # get a rectangle of source image

        if TOF_TYPE=='KINECT':
            rect_dep_image = np.zeros(KINECT_DEP_SZ)
            for i in np.arange(height):
                rect_dep_image[
                (start_y + i) * KINECT_DEP_WID + start_x:(start_y + i) * KINECT_DEP_WID+start_x+width]=dep_image[(start_y + i)*KINECT_DEP_WID+start_x:(start_y+i)*KINECT_DEP_WID+start_x+width]
        elif TOF_TYPE=='NEW_TOF':
            rect_dep_image = np.zeros(NEWTOF_DEP_SZ)
            for i in np.arange(height):
                rect_dep_image[
                (start_y + i) * NEWTOF_DEP_WID + start_x:(start_y + i) * NEWTOF_DEP_WID+start_x+width]=dep_image[(start_y + i)*NEWTOF_DEP_WID+start_x:(start_y+i)*NEWTOF_DEP_WID+start_x+width]
        return rect_dep_image
#if __name__=='__main__':
