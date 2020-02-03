# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:59:02 2020

@author: Sumit
"""


from numba import jit,f8
from numba import cuda,float64,int32
from random import randint
from array import *
import numpy


import numpy as np
matrix = np.loadtxt('adj.csv', usecols=range(100))
print(matrix)
#rows, cols = (5, 10);
#matSize=10
#A = numpy.zeros(shape=(rows,cols))
#B = numpy.zeros(shape=(rows,cols))
#
#f = open("adj.txt","w+");
#for i in range(matSize):
#    for j in range(matSize):
#        num = randint(0,1);
#        if(j==matSize):
#            f.write(str(num));
#        else:
#            f.write(str(num)+" ");
#    f.write("\r")
#f.close();
#
#intCast = lambda l : [float(int(item)) for item in l];
#
#k=l=0;
#fh=open("adj.txt");
#
#for line in fh:
#    if(k<=rows-1):
#        x=line.split(" ");
#        A[k]=intCast(x[:cols]);
#        k+=1;
#    else:
#        x=line.split(" ");
#        B[l]=intCast(x[:cols]);
#        l+=1;
#print(A)
#print(B)
#print(A)
#print(B)

#vector length
N = 100
#number of vectors
NV = 100
#number of threads per block - must be a power of 2 less than or equal to 1024
threadsperblock = 256
#for vectors arranged row-wise
@cuda.jit('void(float64[:,:], float64[:])')
def vec_sum_row(vecs, sums):
    sm = cuda.shared.array(threadsperblock, float64)
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
# load shared memory with vector using block-stride loop
    lid = tid
    sm[lid] = 0
    while lid < N:
        sm[tid] += vecs[bid, lid];
        lid += bdim
    cuda.syncthreads()
# perform shared memory sweep reduction
    sweep = bdim//2
    while sweep > 0:
        if tid < sweep:
            sm[tid] += sm[tid + sweep]
        sweep = sweep//2
        cuda.syncthreads()
    if tid == 0:
        sums[bid] = sm[0]
   
#Aw =np.array(np.random.random((NV, N)), dtype=np.float32)
#print(Aw)        
#rvecs  = np.ones((NV, N), dtype=np.float32)
        
sums   = numpy.zeros(NV, dtype=numpy.float64)
d_rvecs = cuda.to_device(matrix)
d_sums = cuda.device_array_like(sums)
vec_sum_row[NV, threadsperblock](d_rvecs, d_sums)
vv=d_sums.copy_to_host(sums)
print("output is:")
print(vv)

#@jit('f8(f8[:])')
#def sum1d(A):
#    sum = 0.0
#    for i in range(A.shape[0]):
#        sum += A[i]
#    return sum
#d=sum1d(A)
#print(d)


