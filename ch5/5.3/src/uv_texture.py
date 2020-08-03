import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
# src= (
#     (1, -1, -1),
#     (1, 1, -1),
#     (-1, 1, -1),
#     (-1, -1, -1),
#     (2, -2, -1),
#     (2, 2, -1),
#     (-2, -2, -1),
#     (-2, 2, -1)
# )
# dst=(
#     (1, -1, -1),
#     (1, 1, -1),
#     (-1, 1, -1),
#     (-1, -1, -1),
#     (2, -2, -1),
#     (2, 2, -1),
# )
# neigh = NearestNeighbors(n_neighbors=1)
# neigh.fit(dst)
# distances, indices = neigh.kneighbors(src, return_distance=True)

def facet_with_uvcoord(facets,uv_coord):
    facets_uv=np.hstack((facets,uv_coord[facets[:,0]],uv_coord[facets[:,1]],uv_coord[facets[:,2]]))
    return facets_uv
def remove_bad_facets(vertice,facets,pc,threshold=0.02):
    pc_indice=np.arange(len(vertice))
    face_idx=np.arange(len(facets)).astype('bool')
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pc)
    distances, indices = neigh.kneighbors(vertice, return_distance=True)
    mask=distances>threshold
    invalid_idx=pc_indice[mask.flatten()]
    for idx in range(len(facets)):
        if np.any((facets[idx,0] in invalid_idx,facets[idx,1] in invalid_idx,facets[idx,2] in invalid_idx)):
            face_idx[idx]=False
        else:
            face_idx[idx]=True

    facets=facets[face_idx.flatten()]
    return facets

class uv_texture:
    def __init__(self):
        self.uv_coord=[]