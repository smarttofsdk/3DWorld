import numpy as np
import pypoisson
"""
input:pc_rgb(with color)
process:calculate normal vector
        triangulation
        mesh
        add normal vector and facets to pc_rgb
        save file as ply format
"""
def poission_rect(points,normals,depth=7,full_depth=5,scale=1.1,samples_per_node=1,cg_depth=0.0):
    facets,verticle=pypoisson.poisson_reconstruction(points,normals,depth,full_depth,scale,samples_per_node,cg_depth)
    return np.array(facets),np.array(verticle)