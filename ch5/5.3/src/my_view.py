#!/usr/bin/python
#coding=utf-8

import ctypes
import cv2
import pygame
import sys,os
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

N=10
num_tests=100
dim=3
noise_sigma=.01
translation=.1
rotation=.1

def best_fit_transform(A,B):
    assert A.shape==B.shape
    m=A.shape[1]
    centroid_A=np.mean(A,axis=0)
    centroid_B=np.mean(B,axis=0)
    AA=A-centroid_A
    BB=B-centroid_B

    H=np.dot(AA.T,BB)
    U,S,Vt=np.linalg.svd(H)
    R=np.dot(Vt.T,U.T)#?

    if np.linalg.det(R)<0:
        Vt[m-1,:]*=-1
        R=np.dot(Vt.T,U.T)

    t=centroid_B.T-np.dot(R,centroid_A.T)

    T=np.identity(m+1)
    T[:m,:m]=R
    T[:m,m]=t
    return T,R,t
def nearest_neighbor(src,dst):
    assert src.shape==dst.shape
    neigh=NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances,indices=neigh.kneighbors(src,return_distance=True)
    return distances.ravel(),indices.ravel()
def icp(A,B,init_pose=None,max_iterations=20,tolerance=0.001):
    assert A.shape==B.shape
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i

def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
def test_best_fit():
    A=np.random.rand(N,dim)
    time_used=0

    for i in range(num_tests):
        B = np.copy(A)
        #transform
        t=np.random.rand(dim)*translation
        B+=t

        #rotation
        R=rotation_matrix(np.random.rand(dim),np.random.rand()*rotation)
        B=np.dot(R,B.T).T

        #noise
        B+=np.random.randn(N,dim)*noise_sigma

        start_time=time.time()
        T,R1,t1=best_fit_transform(B,A)
        time_used+=time.time()-start_time
        C=np.ones((N,4))
        C[:,0:3]=B
        C=np.dot(T,C.T).T

        assert np.allclose(C[:,0:3],A,atol=6*noise_sigma)
        assert np.allclose(-t1,t,atol=6*noise_sigma)
        assert np.allclose(R1.T,R,atol=6*noise_sigma)

    print('best fit time:{:.3}'.format(time_used/num_tests))
def test_icp():
    A=np.random.rand(N,dim)

    total_time=0
    for i in range(num_tests):
        B=np.copy(A)

        t=np.random.rand(dim)*translation
        B+=t
        R=rotation_matrix(np.random.rand(dim),np.random.rand()*rotation)
        B=np.dot(R,B.T).T

        B+=np.random.randn(N,dim)*noise_sigma
        np.random.shuffle(B)
        #Run icp
        start_time=time.time()
        T,distances,iterations=icp(B,A,tolerance=0.000001)
        total_time=time.time()-start_time
        C=np.ones((N,4))
        C[:,0:3]=np.copy(B)

        C=np.dot(T,C.T).T
        assert np.mean(distances)<6*noise_sigma
        assert np.allclose(T[0:3,0:3].T,R,atol=6*noise_sigma)
        assert np.allclose(-T[0:3,3],t,atol=6*noise_sigma)
    print(T[0:3,0:3].T)
if __name__ == '__main__':
    test_icp()


