import numpy as np
import collections
from simple_3D_viewer import *
import cal_normals
from global_cfg import *
def check_mode(np_points):
    lp=np_points.shape[1]
    mode=0
    if lp==3:
        mode=0
    elif lp==6:
        mode=1
    elif lp==9:
        mode=2
    else:
        print("shape error!")
    return mode
def save_plyfile(ply_file,np_points,np_facets=[],texture_name=[],removeBlackPoints=True):
    """ input : path to plyfile and numpy array of shape (N,6)"""
    has_texture=False
    mode=check_mode(np_points)
    if np_facets!=[] and np.array(np_facets).shape[1]==9:
        has_texture=True
    points =[]
    facets=[]
    idx_p=0
    idx_f=0
    for p in np_points.tolist():
        idx_p+=1
        if mode==0:
            points.append("%f %f %f\n"%tuple(p))
        elif mode==1:
            points.append("%f %f %f %f %f %f\n"%tuple(p))
        elif mode==2:
            points.append("%f %f %f %f %f %f %d %d %d 255\n"%tuple(p))
    if len(np_facets)!=0:
        for f in np_facets.tolist():
            idx_f+=1
            if idx_f == np_facets.shape[0]:
                if has_texture:
                    facets.append("3 %d %d %d 6 %f %f %f %f %f %f" % tuple(f))
                else:
                    facets.append("3 %d %d %d"%tuple(f))
            else:
                if has_texture:
                    facets.append("3 %d %d %d 6 %f %f %f %f %f %f\n" % tuple(f))
                else:
                    facets.append("3 %d %d %d\n"%tuple(f))
    f = open(ply_file,"w")
    if has_texture:
        f.write('''ply
format ascii 1.0
comment TextureFile %s'''%(texture_name))
    else:
        f.write('''ply
format ascii 1.0''')
    if mode==0:
        f.write('''
element vertex %d
property float x
property float y
property float z'''%(len(points)))
    elif mode==1:
        f.write('''
element vertex %d
property float x
property float y
property float z
property float nx
property float ny
property float nz'''%(len(points)))
    elif mode==2:
        f.write('''
element vertex %d
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha'''%(len(points)))
    if len(np_facets)!=0:
        if has_texture:
            f.write('''
element face %d
property list uchar int vertex_indices
property list uchar float texcoord
end_header
%s''' % (len(facets), "".join(points)))
        else:
            f.write('''
element face %d
property list uchar int vertex_indices
end_header
%s'''%(len(facets),"".join(points)))
    else:
        f.write('''
end_header
%s'''%"".join(points))
    f.write("".join(facets))
    f.close()
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
def add_rgb_to_pc(pc,rgb_data):
    rgb_pc=np.hstack((pc,rgb_data))
    return rgb_pc
def points_normals_from(filename):
    array = np.genfromtxt(filename)
    return array[:,0:3], array[:,3:6]

def IO_test(points,colors):
    pass



# A few notes about the PLY format supported by blender and meshlab
# Vertex indices in faces should be 'vertex_indices'
# Per-vertex texture coords should be 's', 't' (and not uv)
# Neither blender nor meshlab does seem to support per-face texcoords (using
# a texcoord list property with 6 entries)
# So we export with duplicate vertices, per-vertex texcoords. We use additional
# properties to map vertices to original SCAPE vertices (scape_vid)


def load_ply(filename, load_bp=False):
    """
    Loads a .ply file.
    Returns verts, faces, faces_uv.
    If the .ply has no texture coordinates, faces_uv is an empty list

    if `load_bp` is True, will try to load property int32 body part and return
    an additional bpids array that gives the body part each vertex belongs too
    """
    with open(filename) as f:
        return load_ply_fileobj(f, load_bp)

def load_ply_fileobj(fileobj, load_bp=False):
    """Same as load_ply, but takes a file-like object"""
    def nextline():
        """Read next line, skip comments"""
        while True:
            line = fileobj.readline()
            assert line != '' # eof
            if not line.startswith('comment'):
                return line.strip()

    assert nextline() == 'ply'
    assert nextline() == 'format ascii 1.0'
    line = nextline()
    #if line.startswith('comment TextureFile')
    assert line.startswith('element vertex')
    nverts = int(line.split()[2])
    #print 'nverts : ', nverts
    assert nextline() == 'property float x'
    assert nextline() == 'property float y'
    assert nextline() == 'property float z'
    line = nextline()
    has_normal=line=='property float nx'
    if has_normal:
        assert nextline() == 'property float ny'
        assert nextline() == 'property float nz'
    line = nextline()
    has_color = line == 'property uchar red'
    if has_color:
        assert nextline() == 'property uchar green'
        assert nextline() == 'property uchar blue'
    return_bp = False
    if load_bp:
        assert line == 'property int32 bpid'
        return_bp = True
        line = nextline()
    elif line == 'property int32 bpid':
        load_bp = True
        return_bp = False
        line = nextline()
    while line.startswith('element face')==False:
        line=nextline()
    assert line.startswith('element face')
    nfaces = int(line.split()[2])
    #print 'nfaces : ', nfaces
    assert nextline() == 'property list uchar int vertex_indices'
    line = nextline()
    has_texcoords = line == 'property list uchar float texcoord'
    if has_texcoords:
        assert nextline() == 'end_header'
    else:
        assert line == 'end_header'

    # Verts
    if load_bp:
        bpids = np.zeros(nverts, dtype=int)
    verts = np.zeros((nverts, 3))
    normals = np.zeros((nverts, 3))
    for i in np.arange(nverts):
        vals = nextline().split()
        verts[i,:] = [float(v) for v in vals[:3]]
        if has_normal:
            normals[i,:]=[float(v) for v in vals[3:6]]
        if load_bp:
            bpids[i] = int(vals[3])
    # Faces
    faces = []
    faces_uv = []
    for i in np.arange(nfaces):
        vals = nextline().split()
        assert int(vals[0]) == 3
        faces.append([int(v) for v in vals[1:4]])
        if has_texcoords:
            assert len(vals) == 11
            assert int(vals[4]) == 6
            faces_uv.append([(float(vals[5]), float(vals[6])),
                             (float(vals[7]), float(vals[8])),
                             (float(vals[9]), float(vals[10]))])
            #faces_uv.append([float(v) for v in vals[5:]])
        else:
            assert len(vals) == 4
    if return_bp:
        return verts, faces, normals,faces_uv, bpids
    else:
        return verts, faces, normals,faces_uv

def read_depth_image(fp):
    if TOF_TYPE=='NEW_TOF':
        frame = np.fromfile(fp, dtype=NEWTOF_DATA_TYPE, count=NEWTOF_DEP_SZ)#new tof np.float32
        frame = frame.copy().astype(np.float32) / 1000.0
    elif TOF_TYPE=='KINECT':
        frame = np.fromfile(fp, dtype=KINECT_DATA_TYPE, count=KINECT_DEP_SZ)
        frame = frame.copy().astype(np.float32)/1000.0 #add this if using kinect
    return frame

def read_image(fp,TYPE='RGB'):
    if TYPE=='RGB':
        frame= np.frombuffer(fp.read(480*640*3), dtype=np.uint8)
        frame= frame.reshape([480,640, 3]).astype(np.uint8)
        # frame_tmp=frame.copy()
        # frame_tmp[:,0]=frame[:,2]
        # frame_tmp[:,2] = frame[:,0]
    elif TYPE=='IR':
        frame = np.frombuffer(fp.read(480 * 640 * 1), dtype=np.uint8)
        frame = frame.reshape([480, 640, 1]).astype(np.uint8)
    return frame
def read_pc(fp):
    # frame= np.frombuffer(fp.read(240*320*4), dtype=np.float32)
    frame = np.fromfile(fp, dtype=NEWTOF_DATA_TYPE, count=NEWTOF_DEP_SZ*4)
    frame= frame.reshape([-1,4]).astype(np.float32)
    return frame[:,0:3]

def read_ir_image(fp,mask=None):
    frame_ir = np.fromfile(fp, dtype=np.int16, count=FRAME_DEP_SZ)
    frame_tmp=np.zeros(frame_ir.shape)
    if mask==None:
        frame_tmp=frame_ir.copy()
    else:
        frame_tmp[mask.flatten()]=frame_ir[mask.flatten()]
    frame_ir = frame_tmp.reshape([KINECT_IR_WID, KINECT_IR_HGT]) / 100
    return frame_ir

points = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)
colors = (
    (100, 0, 0),
    (0, 100, 0),
    (0, 0, 100),
    (0, 100, 0),
    (1, 100, 100),
    (0, 100, 100),
    (100, 0, 0),
    (0, 100, 0),
)
if __name__=="__main__":
    IO_test(points,colors)