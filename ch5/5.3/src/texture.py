import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from depth_cam_tools import *
import sys
import time
import os

textures={}
def LoadTextures(fname):#TODO needs to change to n_tex
    #fname = fname + ".jpg"
    if textures.get(fname) is not None:
        return textures.get(fname)
    texture = textures[fname] = glGenTextures(1)
    image = Image.open(fname)
    ix = image.size[0]
    iy = image.size[1]
    image = image.tobytes("raw", "RGBX", 0, -1)
    # Create Texture
    glBindTexture(GL_TEXTURE_2D, texture)  # 2d texture (x and y size)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    #######para setting#################
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    return texture

def InitGL(Width,Height):
    global surfaces,vertices,uv_coord
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(Width, Height)
    glutInitWindowPosition(500,500)
    window = glutCreateWindow("texture test")
    glutDisplayFunc(DrawObject)
    # glutIdleFunc(DrawObject)
    # glutReshapeFunc(ReSizeGLScene)
    # glutKeyboardFunc(keyPressed)
    #texture=LoadTextures("/usr/share/pixmaps/faces/sunflower.jpg")
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
    glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)  # The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
    glShadeModel(GL_SMOOTH)  # Enables Smooth Color Shading
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # Reset The Projection Matrix
    # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
def ReSizeGLScene(Width, Height):
    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small
        Height = 1
    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
def DrawObject():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear The Screen And The Depth Buffer
    glLoadIdentity()  # Reset The View
    glTranslatef(0.0, 0.0, -5.0)
    glBegin(GL_QUADS)
    pObject=PObject()
    #if pObject.HasTexture:
    glEnable(GL_TEXTURE_2D)
    for n_tex in range(pObject.texs):
        LoadTextures(n_tex)
        tex_facets=pObject.get_facets_of_tex(n_tex)
        for surface in tex_facets:
            #n_tex=pObject.get_texture(surface)
            #LoadTextures(n_tex)
            n_uv=pObject.get_uvcoord(surface)
            idx = 0
            for vert_idx in surface[1:4]:#the vertice index of the facet
                vertice=pObject.p_vertices[vert_idx]
                glTexCoord2f(n_uv[idx][0],n_uv[idx][1])#TODO:
                glVertex3fv(vertice)
                idx += 1
    glEnd()
    glutSwapBuffers()
def keyPressed(*args):
    if args[0] == ESCAPE:
        sys.exit()
####data#####
class PObject:
    def __init__(self):
        self.p_vertices=np.array([])#x y z
        self.p_facets=np.array([])#3 v1 v2 v3 6 uv1x uv1y uv2x uv2y uv3x uv3y n_tex
        #self.p_macterial=[] #not used
        self.HasTexture=False
        self.texs=0
    def get_facets_of_tex(self,n_tex):
        facets_of_tex=[]
        for facet in self.p_facets:
            if n_tex==self.get_texture(facet):
                facets_of_tex.append(facet)
        return np.array(facets_of_tex)
    def get_texture(self,n_facet):
        facet=np.array(self.p_facets)#n*4 the fourth element is the corresponding texture
        n_tex=facet[n_facet,12]#TODO decide the position
        return n_tex
    def get_uvcoord(self,n_facet):#get uv_coord of vertices in n_th facet
        #uv_coords=[]
        #uv_coords.append()
        uv_coords=np.array(self.p_facets[n_facet][5:11])
        # for v_idx in self.p_facets[n_facet][1:3]:
        #     uv_coords.append(self.p_vertices[v_idx][3:4])
        return uv_coords.reshape(-1,2)
    def set_vertices(self,ori_vertice):
        self.p_vertices=np.hstack((ori_vertice))
    def set_facets(self,ori_facets,ori_uv_coords,ori_n_tex):
        self.p_facets=np.hstack((3,ori_facets,6,ori_uv_coords,ori_n_tex))

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (2, -2, -1),
    (2, 2, -1),
    (-2, -2, -1),
    (-2, 2, -1)
)
uv_coord=(
    (0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),
    (0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)
)
surfaces = (
    (0, 1, 2, 3),
    (0, 4, 5, 1),
    (6,7, 3, 2),
    # (4, 5, 1, 0),
    # (1, 5, 7, 2),
    # (4, 0, 3, 6)
)
def texture_test(vertices,surfaces):
    InitGL(640, 480)
    texture = LoadTextures("/usr/share/pixmaps/faces/sunflower.jpg")
    glutMainLoop()


if __name__=='__main__':
    texture_test(vertices,surfaces)