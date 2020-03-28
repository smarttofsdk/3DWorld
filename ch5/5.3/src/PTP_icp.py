import numpy as np
from scipy import spatial
import scipy.spatial.distance as ssd
import cal_normals
# import icp
import pc_trans_tool
'''
point to plane icp algorithm,needs improvement
'''
class NormalCorrespondences:
    def __init__(self, index_map, source_points, target_points, source_normals, target_normals):
        self.source_normals_ = source_normals
        self.target_normals_ = target_normals
        self.source_points_ = source_points
        self.target_points_ = target_points
        self.index_map_=index_map
    def __iter__(self):
        self.iter_count_ = 0

    def next(self):
        if self.iter_count_ >= len(self.num_matches_):
            raise StopIteration
        else:
            return self.source_points_[self.iter_count,:], self.target_points_[self.iter_count,:], self.source_normals_[self.iter_count,:], self.target_normals_[self.iter_count,:]

class PointToPlaneICPSolver:#IterativeRegistrationSolver):
    def __init__(self, sample_size=500, cost_sample_size=1000, gamma=100.0, mu=1e-2):
        self.sample_size_ = sample_size
        self.cost_sample_size_ = cost_sample_size
        self.gamma_ = gamma
        self.mu_ = mu
        #IterativeRegistrationSolver.__init__(self)
    def match(self, source_points, target_points, source_normals, target_normals,dist_thresh=None,norm_thresh=None):
        # dist_thresh = 1.5
        # norm_thresh = 0.2
        import icp
        #distance,match_indices = icp.nearest_neighbor(source_points, target_points)
        #dist_thresh=np.mean(distance)
        dists = ssd.cdist(source_points, target_points, 'euclidean')
        if dist_thresh==None:#TODO find a proper dist_thresh and norm_thresh value
             dist_thresh = np.mean(dists)
        ip = source_normals.dot(target_normals.T) # abs because we don't have correct orientations
        # if norm_thresh==None:
        #     norm_thresh =np.mean(ip[0,:])
        source_ip = source_points.dot(target_normals.T)
        target_ip = target_points.dot(target_normals.T)
        target_ip = np.diag(target_ip)
        target_ip = np.tile(target_ip, [source_points.shape[0], 1])
        abs_diff = np.abs(source_ip - target_ip) # difference in inner products

        # mark invalid correspondences
        invalid_dists = np.where(dists > dist_thresh)
        abs_diff[invalid_dists[0], invalid_dists[1]] = np.inf
        invalid_norms = np.where(ip < norm_thresh)
        abs_diff[invalid_norms[0], invalid_norms[1]] = np.inf

        # choose the closest matches
        match_indices = np.argmin(abs_diff, axis=1)
        match_vals = np.min(abs_diff, axis=1)
        invalid_matches = np.where(match_vals == np.inf)
        match_indices[invalid_matches[0]] = -1
        #import icp
        #_,match_indices=icp.nearest_neighbor(source_points,target_points)
        return NormalCorrespondences(match_indices, source_points, target_points, source_normals, target_normals)
    def register(self, source_point_cloud, target_point_cloud,
                 source_normal_cloud, target_normal_cloud,
                 num_iterations=40, compute_total_cost=True, match_centroids=False,
                 vis=False):
        orig_source_points = source_point_cloud
        orig_target_points = target_point_cloud
        orig_source_normals = source_normal_cloud
        orig_target_normals = target_normal_cloud
        # alloc buffers for solutions
        source_mean_point = np.mean(orig_source_points, axis=0)
        target_mean_point = np.mean(orig_target_points, axis=0)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1]) #init with diff between means
        if match_centroids:
            t_sol[:,0] = target_mean_point - source_mean_point
        # iterate through
        for i in range(num_iterations):
            # subsample points
            source_subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.sample_size_)
            source_points = orig_source_points[source_subsample_inds,:]
            source_normals = orig_source_normals[source_subsample_inds,:]
            target_subsample_inds = np.random.choice(orig_target_points.shape[0], size=self.sample_size_)
            target_points = orig_target_points[target_subsample_inds,:]
            target_normals = orig_target_normals[target_subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T

            # closest points
            corrs = self.match(source_points,target_points,source_normals,target_normals,dist_thresh=None,norm_thresh=0.75)

            # solve optimal rotation + translation
            # valid_corrs = np.where(corrs.index_map_ != -1)[0]
            # sp=corrs.source_points_
            # source_corr_points = corrs.source_points_[valid_corrs,:]
            # target_corr_points = corrs.target_points_[corrs.index_map_[valid_corrs], :]
            # target_corr_normals = corrs.target_normals_[corrs.index_map_[valid_corrs], :]
            # valid_corrs = np.where(corrs.index_map_ != -1)[0]
            # sp = corrs.source_points_
            source_corr_points = corrs.source_points_[:, :]
            target_corr_points = corrs.target_points_[corrs.index_map_[valid_corrs], :]
            target_corr_normals = corrs.target_normals_[corrs.index_map_[valid_corrs], :]

            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                print('No correspondences found')
                break

            # create A and b matrices for Gauss-Newton step on joint cost function
            A = np.zeros([6,6])
            b = np.zeros([6,1])
            Ap = np.zeros([6,6])
            bp = np.zeros([6,1])
            G = np.zeros([3,6])
            G[:,3:] = np.eye(3)

            for i in range(num_corrs):
                s = source_corr_points[i:i+1,:].T
                t = target_corr_points[i:i+1,:].T
                n = target_corr_normals[i:i+1,:].T
                G[:,:3] = skew(s).T
                A += G.T.dot(n).dot(n.T).dot(G)
                b += G.T.dot(n).dot(n.T).dot(t - s)
                Ap += G.T.dot(G)
                bp += G.T.dot(t - s)
            v = np.linalg.solve(A + self.gamma_*Ap + self.mu_*np.eye(6),
                                b + self.gamma_*bp)

            # create pose values from the solution
            R = np.eye(3)
            R = R + skew(v[:3])
            U, S, V = np.linalg.svd(R)
            R = U.dot(V)
            t = v[3:]

            # incrementally update the final transform
            R_sol = R.dot(R_sol)
            t_sol = R.dot(t_sol) + t
        return R_sol, t_sol

def skew(xi):
    S = np.array([[0, -xi[2], xi[1]],
                  [xi[2], 0, -xi[0]],
                  [-xi[1], xi[0], 0]])
    return S
def cal_icp(src,dist,src_normal,dist_normal,init_pose,match_centroids=False):
    src=pc_trans_tool.T_transform(src,init_pose)
    src_normal=pc_trans_tool.T_transform(src_normal,init_pose)
    icp_solver = PointToPlaneICPSolver()
    icp_R, icp_t = icp_solver.register(src, dist, src_normal, dist_normal)
    T=pc_trans_tool.Rt_to_T(icp_R,icp_t)
    return np.dot(init_pose,T)

if __name__=='__main__':
    import time
    src=np.random.randn(1000,3)
    R=np.array([[ 0.99712395,0.02844392,-0.07024798],
 [-0.02557888,0.99881713,0.04135292],
 [ 0.07134113,-0.03943713,0.99667204],])
    t=np.ones((1,3))*0.1
    dist=np.dot(R,src.T).T+t
    #begin=time.time()
    src_normal=cal_normals.compute_feature(src,mean_k=4)
    dist_normal=cal_normals.compute_feature(dist,mean_k=4)
    begin = time.time()
    T=cal_icp(src,dist,src_normal,dist_normal)
    print(T)
    time_cost =time.time()-begin
    print("time cost:",time_cost)