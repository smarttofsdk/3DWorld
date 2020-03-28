cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy,memcpy
from libcpp cimport bool
import ctypes


np.import_array()
LP_c_char = ctypes.POINTER(ctypes.c_char)
LP_LP_c_char = ctypes.POINTER(LP_c_char)
cdef extern from "PoissonRecon_v6_13/src/PoissonReconLib.h":
    cdef int PoissonReconLibMain(int argc, char* argv[])#ctypes.c_int,LP_LP_c_char)#
    cdef vector[double] double_data
    cdef vector[int] int_data
    cdef vector[double] mem_data
    #ctypedef char **c_argv

def poisson_reconstruction(points, normals, depth=8, full_depth=5, scale=1.1, samples_per_node=1.0, cg_depth=0.0,
                            enable_polygon_mesh=False, enable_density=False):
    return _poisson_reconstruction(np.ascontiguousarray(np.float64(points)), np.ascontiguousarray(np.float64(normals)),
                                   depth, full_depth, scale, samples_per_node, cg_depth,
                                   enable_polygon_mesh, enable_density)



cdef _poisson_reconstruction(np.float64_t[:, ::1] points, np.float64_t[:, ::1] normals,
                           int depth=8, int full_depth=5, double scale=1.10, double samples_per_node=1.0, double cg_depth=0.0,
                           bool enable_polygon_mesh=False, bool enable_density=False):


    cdef:
        char **c_argv
        string arg_depth = bytes(str(depth),encoding="utf-8")
        string arg_full_depth = bytes(str(full_depth),encoding="utf-8")
        string arg_scale = bytes(str(scale),encoding="utf-8")
        string arg_samples_per_node = bytes(str(samples_per_node),encoding="utf-8")
        string arg_cg_depth = bytes(str(cg_depth),encoding="utf-8")
        #argv[i] = ctypes.create_string_buffer(enc_arg)

    print('arg_depth',arg_depth)
    int_data.clear()
    double_data.clear()
    mem_data.clear()

    point_nrows, point_ncols = np.shape(points)
    normal_nrows, normal_ncols = np.shape(normals)

    mem_data.resize(point_ncols * point_nrows + normal_ncols * normal_nrows)

    for i in range(point_nrows):
        for j in range(point_ncols):
            mem_data[j +  i*(point_ncols + normal_ncols)] = points[i,j]
            mem_data[j + point_ncols + i *(point_ncols + normal_ncols)] = normals[i,j]


    args = ["PoissonRecon", "--in", "none", "--out", "none", "--depth", str(depth),
                            "--fullDepth",  str(full_depth), "--scale",  str(scale),
                            "--samplesPerNode", str(samples_per_node),
                            "--cgDepth", str(cg_depth)]

    if enable_polygon_mesh:
        args += ["--polygonMesh"]
    if enable_density:
        args += ["--density"]
    a=sizeof(char*)
    b=len(args)
    #print('a=',a,'b=',b)
    c_argv = <char**> malloc(sizeof(char*)*len(args))
    #PoissonReconLibMain.argtypes =(ctypes.c_int,LP_LP_c_char)
    #argv = (LP_c_char * (len(args) + 1))()
    args_arr = [bytearray(x, encoding="utf-8") for idx,x in enumerate(args)]
    #args=shlex.split(str(args))
    #s=bytes[]
    for idx in range(15):
        c_argv[idx]=args_arr[idx]
    try:
        PoissonReconLibMain(len(args),c_argv)
        for i in range(15):
            print(c_argv[i])
        #print(c_argv[0],'\n',c_argv[1],'\n',c_argv[2],c_argv[3],'\n',c_argv[4],'\n',c_argv[5],'\n',c_argv[6],'\n',c_argv[7],'\n',c_argv[8],'\n')
        print('succed')
    finally:
        free(c_argv)


    face_cols, vertex_cols = 3, 3
    face_rows = int_data.size() / face_cols
    vertex_rows = double_data.size() / vertex_cols

    cdef int *ptr_faces = &int_data[0]
    cdef double *ptr_vertices = &double_data[0]

    faces = np.zeros((int(face_rows*face_cols),), dtype=np.int32 )
    vertices = np.zeros((int(vertex_rows*vertex_cols),), dtype=np.float64)

    for i in range(int(face_rows*face_cols)):
        faces[i] = ptr_faces[i]

    for i in range(int(vertex_rows*vertex_cols)):
        vertices[i] = ptr_vertices[i]

    int_data.clear()
    double_data.clear()

    return faces.reshape(int(face_rows),int(face_cols)), vertices.reshape(int(vertex_rows),int(vertex_cols))


