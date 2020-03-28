import numpy as np

class PC:   # frame_idx x y x n_x n_y n_z rgb_x rgb_y rgb_z
    def __init__(self,pc):
        self.pc=pc
        self.len=self.pc.shape[0]
        #if self.pc.shape[1]
        self.pc_n=[]#pc with normal
        #self.pc_n_c=[] #pc with normal and color
    def get_pc(self,pc_idx=None,mode=None):
        if pc_idx==None:
            pc=self.pc[:,0:4]
        else:
            pc=self.pc[pc_idx,0:4]
        if mode=='NO_FRAME_IDX':
            pc=pc[:,1:4]
        return pc
    def get_normal(self,n_idx=None):
        assert self.pc.shape[1]>3
        if n_idx==None:
            return self.pc[:,4:7]
        else:
            return self.pc[n_idx,4:7]

    def get_color(self,color_idx=None):
        assert self.pc.shape[1]>6
        if color_idx==None:
            return self.pc[:,7:10]
        else:
            return self.pc[color_idx,7:10]

    def add_normal(self,normals):
        assert normals.shape[0]==self.pc.shape[0]
        pc_tmp=self.get_pc()
        self.pc=np.hstack((pc_tmp,normals))

    def add_color(self,color):
        self.pc=np.hstack((self.pc,color))
        # if self.pc
        # self.pc_n=

if __name__=='__main__':
    pc_rand=np.random.rand(4,3)
    my_pc=PC(pc_rand)
    print(my_pc.get_pc(3))
    my_pc.add_normal(pc_rand)
    print(my_pc.get_normal())
    my_pc.add_color(pc_rand)
    print(my_pc.get_normal())
    print(my_pc.get_pc(3))