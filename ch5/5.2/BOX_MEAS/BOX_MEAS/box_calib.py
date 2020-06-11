#!/usr/bin/python3
# coding=utf-8

import numpy as np
import os

#import pylab as plt


## 从特定目录得到所有文件名
def gen_fname_list(rootdir):
    flist=[]
    for pname,dnames,fnames in os.walk(rootdir):      
        flist+=[ os.path.join(pname,fname) for fname in fnames]
    
    flist_filter=[]
    for fname in flist:
        idx=fname.rfind('.')
        if idx<0: continue
        if fname[idx+1:]=='csv' or fname[idx+1:]=='txt' or fname[idx+1:]=='log':
            flist_filter.append(fname)
    
    print('[INF] gen_fname_list(%s) return:'%rootdir)
    for fname in flist_filter:
        print('[INF]    %s'%fname)
    print()
    return flist_filter


## 从原始测量数据计算最终测量结果
class meas_conv_c:
    ## 如果输入参数：fname_calib_data=None的话，从fname_param加载校准参数
    ##        否者根据fname_calib_data读入原始测量数据，计算校准参数后存入fname_param
    def __init__(self, fname_calib_data=None, fname_param='./config/calib_param.txt', fname_box_size='./config/box_size.txt'):
        # 加载盒子尺寸数据
        self.load_box_size(fname_box_size)

        # 加载校准测量数据，计算校准参数
        if fname_calib_data is not None:
            self.calib(fname_calib_data)
            self.save_calib_param(fname_param)
        elif fname_param is not None: # 直接加载校准参数
            self.load_calib_param(fname_param)

        return


    # 加载校准测量数据，计算校准参数
    def calib(self,fname):
        fname_list=gen_fname_list(fname)

        self.meas_data={}
        
        # 基于校准件数据文件计算校准模型参数
        for fname in fname_list:
            print('[INF] calibation file',fname)
            
            # 提取盒子ID
            # 文件名为xxx_id.xxx
            idx1=fname.rfind('.')
            idx2=fname.rfind('_')
            
            if idx1<0 or idx2<0 or idx2>=idx1:
                print('[ERR] cannot find box ID from file name %s'%fname)
                #import IPython
                #IPython.embed()
                continue
            box_id=eval(fname[idx2+1:idx1])
            print('[INF] box id:',box_id)
            
            # 提取测量数据
            data=self.get_data_from_calib_log(fname)
                
            if box_id in self.meas_data:
                self.meas_data[box_id]+=data    # 和之前的数据合并
            else:
                self.meas_data[box_id]=data     # 加入测量数据
            
        # 执行校准运算
        self.calc_calib_param()

    def save_calib_param(self, fname='./config/calib_param.txt'):
        fp=open(fname,'wt')
        s='%.12f,%.12f,%.12f,%.12f\n'%(self.k2, self.k3, self.k4, self.k5)
        fp.write(s)
        fp.close()
        print('[INF] write calib param to file:',fname)
        print('[INF]    ',s)
        return

    def load_calib_param(self, fname='./config/calib_param.txt'):
        print('[INF] load calib param to file:',fname)
        fp=open(fname,'rt')
        for ln in fp:
            ln=ln.strip()
            if len(ln)==0: continue
            print('[INF]    ',ln)
            self.k2,self.k3,self.k4,self.k5=eval(ln)
        print('[INF]    self.k2:',self.k2)
        print('[INF]    self.k3:',self.k3)
        print('[INF]    self.k4:',self.k4)
        print('[INF]    self.k5:',self.k5)
        return

            
    def load_box_size(self,fname):
        self.box_size={}
        print('[INF] loading standard box size from file',fname)
        fp=open(fname,'rt')
        for ln in fp:
            ln=ln.strip()
            if len(ln)==0: continue
             
            box_id,box_wid,box_len,box_hgt=eval(ln)
            if box_wid>box_len: box_wid,box_len=box_len,box_wid
            self.box_size[box_id]=(box_wid,box_len,box_hgt)
        fp.close()
        for box_id in self.box_size:
            print('[INF]    id:',box_id,'size:',self.box_size[box_id])
        
        

    ## 从log文件提取测量数据
    # TODO：增加数据清洗
    def get_data_from_calib_log(self,fname):
        fp=open(fname,'rt')
        data=[]
        for ln in fp:
            ln=ln.strip()
            if len(ln)==0: continue
             
            frame_cnt,box_wid,box_len,box_hgt=eval(ln)
            if box_wid>box_len: box_wid,box_len=box_len,box_wid
            
            data.append([frame_cnt,box_wid,box_len,box_hgt])
            #print('[INF]    ',frame_cnt,',',box_wid,',',box_len,',',box_hgt)
        fp.close()
        
        # 数据清洗，去除最大最小
        if len(data)>6:
            idx=np.argmin([data[n][1] for n in range(len(data))])
            data.pop(idx)
            idx=np.argmax([data[n][1] for n in range(len(data))])
            data.pop(idx)

            idx=np.argmin([data[n][2] for n in range(len(data))])
            data.pop(idx)
            idx=np.argmax([data[n][2] for n in range(len(data))])
            data.pop(idx)
        
            idx=np.argmin([data[n][3] for n in range(len(data))])
            data.pop(idx)
            idx=np.argmax([data[n][3] for n in range(len(data))])
            data.pop(idx)
        
        print('[INF]    length of measure data:',len(data))
        return data


    ## 参数拟合
    # TODO：增加多种拟合公式
    #       增加正则化，防止过拟合
    #       用非二次代价的代价函数
    #       水平尺寸加入深度数据参与校准
    def calc_calib_param(self):
        # wid_corr =k2*wid+k3
        # hgt_corr =k4*len+k5
        meas=[]
        ref=[]
        for idx,data in self.meas_data.items():

            for _,box_wid,box_len,box_hgt in data:
                meas.append([box_wid, 1])
                meas.append([box_len, 1])
                
                box_wid_r,box_len_r=self.box_size[idx][0],self.box_size[idx][1]
                if box_wid_r>box_len_r: box_wid_r,box_len_r=box_len_r,box_wid_r

                ref.append(box_wid_r)
                ref.append(box_len_r)
        
        mat_meas=np.array(meas)
        vec_ref =np.array(ref).reshape(len(ref),1)
        self.k2,self.k3=self.calc_linear_fit(mat_meas, vec_ref).flatten()
        
        #plt.clf()
        #plt.plot(vec_ref,mat_meas[:,0],'.r')
        #plt.title('XY size vs reference')
        #plt.xlabel('reference')
        #plt.ylabel('XY size')
        #plt.show()
        
        meas=[]
        ref=[]
        for idx,data in self.meas_data.items():
            for _,_,_,box_hgt in data:
                meas.append([box_hgt, 1])
                ref.append(self.box_size[idx][2])
        mat_meas=np.array(meas)
        vec_ref =np.array(ref).reshape(len(ref),1)
        
        #plt.clf()
        #plt.plot(vec_ref,mat_meas[:,0],'.r')
        #plt.title('Z size vs reference')
        #plt.xlabel('reference')
        #plt.ylabel('Z size')
        #plt.show()

        self.k4,self.k5=self.calc_linear_fit(mat_meas, vec_ref).flatten()
        print('[INF] calibation param')
        print('[INF]    k2:',self.k2)
        print('[INF]    k3:',self.k3)
        print('[INF]    k4:',self.k4)
        print('[INF]    k5:',self.k5)


    ## 线性模型拟合
    def calc_linear_fit(self,mat,vec):
        print('[INF] meas_conv_c.calc_linear_fit(mat,vec)')
        print('[INF]    mat.shape:',mat.shape)
        print('[INF]    vec.shape:',vec.shape)
        
        param=np.dot(np.linalg.pinv(np.dot(mat.T,mat)),np.dot(mat.T,vec))
        err=np.dot(mat,param)-vec
        print('[INF]    fit error:',np.sqrt((err**2).mean()))
        return param
        
    
    ## 由原始测量数据输出尺寸结果
    def conv(self,box_wid,box_len,box_hgt):
        #print('[INF] meas_conv_c.conv(',box_wid,',',box_len,',',box_hgt,')')

        box_wid_corr =self.k2*box_wid+self.k3
        box_len_corr =self.k2*box_len+self.k3
        box_hgt_corr =self.k4*box_hgt+self.k5
        
        return box_wid_corr, box_len_corr, box_hgt_corr

  
