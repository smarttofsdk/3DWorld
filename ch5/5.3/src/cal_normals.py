# import numpy as np
# import pcl
# cloud=pcl.PointCloud(np.random.rand(20, 3).astype('float32'))
#
# pcl.save(cloud,'./bunny.ply')
# tree = cloud.make_kdtree()
# ne = cloud.make_NormalEstimation()
# ne.set_SearchMethod (tree)
# ne.set_RadiusSearch(5)
# cloud_with_normals = ne.compute()
#用pcl自带的计算法向量的函数到这步就死机，原因不明，重写一个
# print('>> Done: ' + ' ms\n')
import numpy as np
from sklearn.neighbors import KDTree
def compute_feature(pc,mean_k):
    view_point=np.array([0,0,0],dtype='float64')
    z_mean=np.mean(pc[:, 2])
    #view_point = np.array([0, 0, a])
    dtype = [('normal_x', 'f8'), ('normal_y', 'f8'), ('normal_z', 'f8'), ('curvature', 'f8')]
    params = np.empty((len(pc),), dtype=dtype)

    kDTree = KDTree(pc,leaf_size=10)
    _,nn_indices = kDTree.query(pc[:, :], k=mean_k)
    for idx in range(len(pc)):
        #nn_indices, _ = self._search_for_neighbours(idx, self._search_parameter)
        plane_param, curvature = compute_point_normal(pc, nn_indices[idx])
        # flip_normal_towards_viewpoint
        if np.dot(view_point - pc[idx], plane_param[:3])<0:
            params[idx] = -plane_param[0], -plane_param[1], -plane_param[2], \
                            curvature
        else:
            params[idx] = plane_param[0], plane_param[1], plane_param[2], \
                            curvature
    params=np.array(params.tolist()).reshape(-1,4)
    output=params[:,0:3]
    #params=params.reshape((-1,4))
    #output =pcl.PointCloud(params[:,0:3].astype('float32'))
    #output.copy_metadata(self._input)
    return output
def re_compute_normal(pc_with_n,mean_k):
    pc=pc_with_n[:,0:3]
    ori_normals=pc_with_n[:,3:6]
    z_mean=np.mean(pc[:, 2])
    #view_point = np.array([0, 0, a])
    dtype = [('normal_x', 'f8'), ('normal_y', 'f8'), ('normal_z', 'f8'), ('curvature', 'f8')]
    params = np.empty((len(pc),), dtype=dtype)
    kDTree = KDTree(pc,leaf_size=10)
    _,nn_indices = kDTree.query(pc[:, :], k=mean_k)
    for idx in range(len(pc)):
        #nn_indices, _ = self._search_for_neighbours(idx, self._search_parameter)
        plane_param, curvature = compute_point_normal(pc, nn_indices[idx])
        # flip_normal_towards_viewpoint
        if np.dot(ori_normals[idx], plane_param[:3])<0:
            params[idx] = -plane_param[0], -plane_param[1], -plane_param[2], \
                            curvature
        else:
            params[idx] = plane_param[0], plane_param[1], plane_param[2], \
                            curvature
    params=np.array(params.tolist()).reshape(-1,4)
    output=params[:,0:3]
    #params=params.reshape((-1,4))
    #output =pcl.PointCloud(params[:,0:3].astype('float32'))
    #output.copy_metadata(self._input)
    return output

def compute_point_normal(cloud, indices):
    if len(indices) < 3:
        raise ValueError('not enough points for computing normal')

    # Computing normal vector and curvature using PCA
    covariance_matrix, xyz_centroid = compute_mean_and_covariance_matrix(cloud, indices)
    eigen_value, eigen_vector = np.linalg.eigh(covariance_matrix)
    smallest = np.argmin(eigen_value)

    plane_parameters = eigen_vector[:, smallest].tolist()
    plane_parameters.append(-np.dot(plane_parameters + [0], xyz_centroid))
    eigen_sum = np.sum(eigen_value)
    if eigen_sum != 0:
        curvature = np.abs(eigen_value[smallest] / eigen_sum)
    else:
        curvature = 0

    return plane_parameters, curvature

def compute_mean_and_covariance_matrix(cloud, indices=None, bias=True):

    if indices is not None:
        cloud = cloud[indices]
    # Filter invalid points
    # cloud = cloud[~np.isnan(np.sum(cloud.xyz, axis=1))].data
    #cloud = cloud.data

    # a bit faster than np.cov if compute centroid and covariance at the same time
    # testing shows this method is faster at ratio about 30%, however precision has lost a bit
    accu = dict()
    cloudx = cloud[:,0].T
    cloudy = cloud[:,1].T
    cloudz = cloud[:,2].T
    coef_xx = np.mean(cloudx * cloudx)
    coef_xy = np.mean(cloudx * cloudy)
    coef_xz = np.mean(cloudx * cloudz)
    coef_yy = np.mean(cloudy * cloudy)
    coef_yz = np.mean(cloudy * cloudz)
    coef_zz = np.mean(cloudz * cloudz)
    coef_x = np.mean(cloudx)
    coef_y = np.mean(cloudy)
    coef_z = np.mean(cloudz)
    centroid = [coef_x, coef_y, coef_z, 1] # homogeneous form

    if not bias:
        for key in accu:
            accu[key] *= len(cloud)/(len(cloud) - 1)
    cov = np.zeros((3, 3))
    cov[0, 0] = coef_xx - coef_x**2
    cov[1, 1] = coef_yy - coef_y**2
    cov[2, 2] = coef_zz - coef_z**2
    cov[0, 1] = cov[1, 0] = coef_xy - coef_x*coef_y
    cov[0, 2] = cov[2, 0] = coef_xz - coef_x*coef_z
    cov[1, 2] = cov[2, 1] = coef_yz - coef_y*coef_z

    return cov, centroid
def add_normal(pc):
    if pc.shape[0]>=10:
        k=10
    else:
        k=pc.shape[0]
    pc_normal=compute_feature(pc,k)
    pc_with_n=np.hstack((pc,pc_normal))
    return pc_with_n,pc_normal

def test():
    #pc=pcl.PointCloud(np.identity(3).astype('float32'))
    pc=np.random.randn(5,3)
    #pc=pcl.PointCloud(pc.astype('float32'))

    print(pc)
    #pc_normal=compute_feature(pc,mean_k=2)
    pc_with_n,pc_normal=add_normal(np.array(pc))
    #pc_with_n=pcl.PointCloud(pc_with_n.astype('float32'))
    #pcl.save(pc_with_n,'a.ply')
    print(pc_with_n)
if __name__=='__main__':
    test()