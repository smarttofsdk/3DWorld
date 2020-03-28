import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from depth_cam_tools import *
import cv2
ax = ay = 0.0  # 点云旋转角度
cz = 1.0  # 点云旋转中心
mz = 0  # 点云观察点位置
mouse_down = False
mouse_x = mouse_y = 0
update_trans_mat=False
autorotate=False
wireframe_mode=False
pure_points_mode=True
mouse_button=[False,False,False]
def draw(vertices,surfaces,colors=[]):
    uv_coord=None
    if vertices.shape[1]==5:
        vertices=vertices[:,0:3]
        uv_coord=vertices[:,3:5]
    if pure_points_mode:
        glColor3f(1, 1, 1)
        glPointSize(3.0)
        glBegin(GL_POINTS)
        for i in range(len(vertices)):
            glVertex3f(vertices[i][0],vertices[i][1],vertices[i][2])
    elif wireframe_mode:
        edges=edge_from_surface(surfaces)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
    else:#show 3D model
        color = ((0.5, 0.5, 0.5),
                 (0.25, 0.25, 0.25),
                 (0.6, 0.6, 0.6),
                 (0.8, 0.8, 0.8))
        if len(surfaces[0])==3:
            glBegin(GL_TRIANGLES)
        elif len(surfaces[0]) == 4:
            glBegin(GL_QUADS)
        else:
            glBegin(GL_POLYGON)
        for surface in surfaces:
            x = 0
            if uv_coord != None:
                for vertex in surface:
                    glTexCoord2f(uv_coord[vertex])
                    glVertex3fv(vertices[vertex])
                    x += 1
            else:
                for vertex in surface:
                    glVertex3fv(vertices[vertex])
                    x += 1
    glEnd()


def display_mesh(mesh_data):
    global ax,ay,cz,mz,mouse_down,mouse_x,mouse_y,update_trans_mat,mouse_button,autorotate,wireframe_mode,pure_points_mode
    vertices = mesh_data[0]
    surfaces = mesh_data[1]
    colors=mesh_data[2]
    #normals=mesh_data[3]
    wireframe_mode = False
    autorotate = False

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -3)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    while True:
            # update_trans_mat = False
        T=np.identity(4)
        evt, param = poll_evt()
        if evt is not None:
            print(evt)
            if evt == 'md0':
                mouse_down = True
                mouse_x, mouse_y = param[0], param[1]
                print(mouse_x,mouse_y)
            elif evt == 'mu0':
                mouse_down = False
            elif evt == 'mm':
                if mouse_down:
                    dx = param[0] - mouse_x
                    dy = param[1] - mouse_y
                    mouse_x, mouse_y = param[0], param[1]
                    ax = -dy / 200.0
                    ay = -dx / 200.0
                    update_trans_mat = True
            elif evt == 'mw0':
                mz += 0.1
                glTranslatef(0, 0, 0.1)
                update_trans_mat = True
            elif evt == 'mw1':
                mz -= 0.1
                glTranslatef(0, 0, -0.1)
                print(mz)
                update_trans_mat = True

                # 根据鼠标动作更新显示的3D内容
        if update_trans_mat:
            # T = pc_movz(-cz)
            T = np.dot(T, pc_rotx(ax))
            T = np.dot(T, pc_roty(ay))
            # T = np.dot(T, pc_movz(cz))
            # T = np.dot(T, pc_movz(mz))
            #print(T)
            update_trans_mat=False
        vec_n=np.zeros((3,1))
        cv2.Rodrigues(T[0:3,0:3],vec_n)
        theta=np.linalg.norm(vec_n)
        #print('theta=',theta)
        glRotatef(np.rad2deg(theta),vec_n[0],vec_n[1],vec_n[2])
        #glTranslatef(T[0:3,3].T)
        if autorotate:
             glRotatef(0.3, 1, 3, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw(vertices,surfaces,colors)
        pygame.display.flip()
        pygame.time.wait(10)
def edge_from_surface(surfaces):
    edge=[]
    edges=[]
    surfaces=np.sort(surfaces,axis=1)
    for surface in surfaces:
        for i in range(len(surface)-1):
            for j in range(i+1,len(surface)):
                if surface[i]<surface[j]:#surface[i] is the vertex of this surface
                    edge.append([surface[i],surface[j]])
    edge=np.array(edge).reshape(-1,2)
    edge=edge[np.lexsort(edge[:,::-1].T)]
    for i in range(len(edge)-1):
        if not np.all(edge[i]==edge[i+1]):
            edges.append(edge[i])
    return np.reshape(edges,(-1,2))
def poll_evt():
    #for event in pygame.event.get():  # User did something
    global autorotate,wireframe_mode,pure_points_mode
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            print(event.key)
            if event.key == 27:
                pygame.quit()
                quit()
            if event.key == 113:  # Q
                glRotatef(10, 0, 0, 1)
            if event.key == 101:  # E
                glRotatef(10, 0, 0, -1)
            if event.key == 119:  # W
                glRotatef(10, 1, 0, 0)
            if event.key == 115:  # S
                glRotatef(10, -1, 0, 0)
            if event.key == 97:  # A
                glRotatef(10, 0, 1, 0)
            if event.key == 100:  # D
                glRotatef(10, 0, -1, 0)
            if event.key == 32:  # SPACE
                if autorotate:
                    autorotate = False
                else:
                    autorotate = True
            if event.key == 109:  # M
                if wireframe_mode:
                    wireframe_mode = False
                else:
                    wireframe_mode = True
            if event.key == 112:  # P
                if pure_points_mode:
                    pure_points_mode = False
                else:
                    pure_points_mode = True
        # if event.type == pygame.QUIT:  # If user clicked close
        #     return 'quit', ''
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == 27 or event.key == ord('q'):
        #         return 'quit', ''
        #     elif event.key == ord('s'):
        #         return 'key', 's'
        #     else:
        #         return 'kd', event.key
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pressed_array = pygame.mouse.get_pressed()
            if pressed_array[0] == 1:
                mouse_button[0] = True
                return 'md0', pygame.mouse.get_pos()
            if pressed_array[1] == 1:
                mouse_button[1] = True
                return 'md1', pygame.mouse.get_pos()
            if pressed_array[2] == 1:
                mouse_button[2] = True
                return 'md2', pygame.mouse.get_pos()
            if event.button == 5:
                return 'mw0', pygame.mouse.get_pos()
            if event.button == 4:
                return 'mw1', pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            pressed_array = pygame.mouse.get_pressed()
            if pressed_array[0] == 0 and mouse_button[0] == True:
                mouse_button[0] = False
                return 'mu0', pygame.mouse.get_pos()
            if pressed_array[1] == 0 and mouse_button[1] == True:
                mouse_button[1] = False
                return 'mu1', pygame.mouse.get_pos()
            if pressed_array[2] == 0 and mouse_button[2] == True:
                mouse_button[2] = False
                return 'mu2', pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEMOTION:
            pos = pygame.mouse.get_pos()
            x, y = pos[0], pos[1]
            return 'mm', pygame.mouse.get_pos()

        else:
            return None, ''
    return None, ''
vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)
surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6)
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
)
def test():
    #mesh_data=[]
    mesh_data=(tuple(vertices),tuple(surfaces), tuple(colors))
    display_mesh(mesh_data)

if __name__=='__main__':
    test()