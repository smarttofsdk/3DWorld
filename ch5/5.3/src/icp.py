import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import math
import random
import numpy.ma as ma
import pc_trans_tool
class icp():
    @staticmethod
    def best_fit_transform(A, B):
        assert A.shape == B.shape
        # get number of dimensions
        m = A.shape[1]
        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        #print(Vt)
        R = np.dot(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)
        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)
        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    @staticmethod
    def nearest_neighbor(src, dst):
        #assert src.shape == dst.shape
        neigh = NearestNeighbors(n_neighbors=1 )
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    @staticmethod
    def icp(A, B, init_pose=None,max_iterations=40,tolerance=0.00000001,mode='PTPOINT'):
        #assert A.shape == B.shape
        # get number of dimensions
        m = A.shape[1]
        data_len=A.shape[0]
        #mask=np.ones(data_len).astype(bool)
        # make points homogeneous, copy them to maintain the originals
        dst_normal=B[:,3:6]

        src = np.ones((m+1,A.shape[0]))
        dst = np.ones((m+1,B.shape[0]))
        dst_normal = np.ones((m + 1, B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B[:,0:3].T)#
        dst_normal[:m, :] = np.copy(B[:, 3:6].T)
        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)
        prev_error = 0
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            src_indices,indices,mean_error=icp.get_coor(src[:m,:].T, dst[:m,:].T,dst_normal[:m,:].T, mode)
            #compute the transformation between the current source and nearest destination points

            if mode=='PTPOINT':#point to point
                T,_,_ = icp.best_fit_transform(src[:m,src_indices[:]].T, dst[:m,indices[:]].T)#pan.17/10/23 add -5
            elif mode=='PTPLANE':#point to plane
                T=icp.PTP_icp_process(src[:m,src_indices[:]].T, dst[:m,indices[:]].T,dst_normal[:m,indices[:]].T)
            #T, _, _ = best_fit_transform(src[:m,:].T, dst[:m, indices[:]].T)
            elif mode=='RANSAC':#ransac + point to point
                T=icp.ransac_icp(src[:,src_indices[:]], dst[:,indices[:]],sample_points=int(len(src_indices)*0.8),max_error=mean_error*0.5)
            else:
                raise NameError('no such icp mode!')
            # update the current source
            src = np.dot(T, src)
            if np.abs(prev_error - mean_error) < tolerance:# and i>int(max_iterations/2):
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = icp.best_fit_transform(A, src[:m,:].T)

        return T

    @staticmethod
    def get_coor(src,dst,dst_normal,mode):#get correspondence points
        data_len=len(src)
        if mode=='PTPOINT' or mode=='PTPLANE':#TODO use different matching method
            distances, indices = icp.nearest_neighbor(src,dst)
            # check error
            mean_error = np.mean(distances)
            # remove bad matcher (added by pan 10.29)
            src_indices = np.arange(data_len)
            mask = distances < (mean_error)

            src_indices = src_indices[mask.flatten()]
            indices = indices[mask.flatten()]
        elif mode=='PTPLAIN':
            pass
        return src_indices,indices,mean_error

    @staticmethod
    def ransac_icp(src,dst,sample_points=500,max_error=0.002,max_iterations=5):
        m = src.shape[0]-1
        data_len = src.shape[1]
        i=0
        k=100
        I_max=0
        while i<=max_iterations:
            #ran_idx=np.random.sample(indices)#随机选取w个点
            src_indice=np.arange(data_len)
            #map_indix=np.hstack((np.array([src_indice]).T,np.array([src_indice]).T))
            ran_idx=np.array(random.sample(list(src_indice),sample_points))
            #T, _, _ = best_fit_transform(src[:m, ran_idx[:,0]].T, dst[:m, ran_idx[:,1]].T)
            T, _, _ = icp.best_fit_transform(src[:m, ran_idx[:]].T, dst[:m, ran_idx[:]].T)
            #T, _, _ =icp(pc_ref[0:icp_len, :], pc[0:icp_len, :], init_pose=np.linalg.inv(init_pose)
            #check error and collect inliners
            src = np.dot(T, src)
            #inlier_idx=np.abs(src[:m,:].T - dst[:m,:].T)<max_error  #TODO try other error defination
            inlier_idx=icp.Iliner_idx(src[:m,:].T,dst[:m,:].T,max_error)
            I_count=inlier_idx.sum()
            if I_count>I_max:
                I_max=I_count
                best_inlier_idx=inlier_idx
            i+=1
        I_T, _, _ = icp.best_fit_transform(src[:m, best_inlier_idx[:]].T, dst[:m, best_inlier_idx[:]].T)
        return I_T

    @staticmethod
    def Iliner_idx(src,dst,max_error):
        idx=np.zeros(len(src),dtype="bool")
        for i in range(len(src)):
            if np.linalg.norm(src[i,:]-dst[i,:])<max_error:
                idx[i]=True
            else:
                idx[i]=False
        return idx

    @staticmethod
    def PTP_icp_process(source_corr_points,target_corr_points,target_corr_normals,gamma=100.0, mu=1e-2):
        num_corrs=len(source_corr_points)
        A = np.zeros([6, 6])
        b = np.zeros([6, 1])
        Ap = np.zeros([6, 6])
        bp = np.zeros([6, 1])
        G = np.zeros([3, 6])
        G[:, 3:] = np.eye(3)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1])  # init with diff between means
        for i in range(num_corrs):
            s = source_corr_points[i:i + 1, :].T
            t = target_corr_points[i:i + 1, :].T
            n = target_corr_normals[i:i + 1, :].T
            G[:, :3] = icp.skew(s).T
            A += G.T.dot(n).dot(n.T).dot(G)
            b += G.T.dot(n).dot(n.T).dot(t - s)
            Ap += G.T.dot(G)
            bp += G.T.dot(t - s)
        v = np.linalg.solve(A + gamma* Ap + mu* np.eye(6),
                            b + gamma* bp)

        # create pose values from the solution
        R = np.eye(3)
        R = R + icp.skew(v[:3])
        R=np.array(R,dtype=np.float)
        U, S, V = np.linalg.svd(R)
        R = U.dot(V)
        t = v[3:]

        # incrementally update the final transform
        R_sol = R.dot(R_sol)
        t_sol = R.dot(t_sol) + t
        return pc_trans_tool.Rt_to_T(R_sol,t_sol)

    @staticmethod
    def cal_icp(pc,pc_ref,init_pose):#TODO apply ransac algorithm
        icp_T=icp.icp(pc,pc_ref,init_pose,tolerance=0.00000001,mode='PTPLANE')
        R=icp_T[0:3,0:3]
        t=icp_T[0:3,3]
        #print(icp_T,'\n',R,'\n',t,'\n\n')
        return icp_T,R,t

    @staticmethod
    def skew(xi):
        S = np.array([[0, -xi[2], xi[1]],
                      [xi[2], 0, -xi[0]],
                      [-xi[1], xi[0], 0]])
        return S