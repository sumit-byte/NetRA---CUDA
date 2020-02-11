# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:30:45 2020

@author: Acer
"""
from numba import jit,f8
from numba import cuda,float64,int32
from random import randint
from array import *
import numpy
from timeit import default_timer as timer
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


#import constant_vars as const
import numpy as np

# NUMBER OF KERNELS TO BE USED : DEFAULT VALUE = 16
k_num = 16;
file_name = 'adj.csv';

# READ ADJACENCY MATRIX
matrix = np.loadtxt(file_name);

# DIMENSIONS OF ROWS AND COLUMNS
nRows = matrix.shape[0];
nCols = matrix.shape[1];
        
class matArr:
    def __init__(self):
        self.matrix=[];
    def addRow(self,row):
        self.matrix.append(row);
    def getMat(self):
        return self.matrix;

class genMatrices:
    
    # GET GENERATED ARRAY OF MATRICES
    def getMatArray(self):
        # DEFINE NUMBER OF SPLITS ON THE ADJACENCY MATRIX
        rowsPerSplit = nRows//k_num;
        adRows = nRows-(rowsPerSplit*k_num);

        # Define empty matrices for K_NUM kernels
        matList = [matArr() for i in range(k_num)];

        # Populate Matrices with rowsPerSplit number of rows
        counter = 0;
        for matNum in range(k_num):
            for _ in range(rowsPerSplit):
                if(counter<nRows):
                    matList[matNum].addRow(matrix[counter]);
                    counter+=1;

        # Populate remaining rows into the last matrix           
        for i in range(counter,nRows):
            matList[k_num-1].addRow(matrix[i]);
            
        return matList;
        
genMat = genMatrices();
arr=genMat.getMatArray();
threadsperblock = 512
#print(arr[0].getMat())
@cuda.jit('void(float64[:,:],float64[:])')
def vec_sum_row(vecs,sums):
    sm = cuda.shared.array(threadsperblock, float64)
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    
   
# load shared memory with vector using block-stride loop
    lid = tid
    sm[lid] = 0
    while lid < nCols:
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
        
        
        
        
        
@profile
def testProfile():
          
    #rvecs  = np.ones((NV, N), dtype=np.float32)
    #n=0
    for i in range(0,k_num):        
        sums   = numpy.zeros(6, dtype=numpy.float64)
        #sums1   = numpy.zeros(NV, dtype=numpy.float64)
       # sums2   = numpy.zeros(NV, dtype=numpy.float64)
        if(i==k_num-1):
            sums1   = numpy.zeros(len(arr[i].getMat()), dtype=numpy.float64)
            d_rvecs1 = cuda.to_device(arr[i].getMat())
            d_sums1 = cuda.device_array_like(sums1)
            vec_sum_row[len(arr[i].getMat()), threadsperblock](d_rvecs1,d_sums1)
            ss=d_sums1.copy_to_host(sums1)
            print(ss)
    #    print(matrix)
        else:
            d_rvecs = cuda.to_device(arr[i].getMat())
            d_sums = cuda.device_array_like(sums)
            vec_sum_row[len(arr[i].getMat()), threadsperblock](d_rvecs,d_sums)
            vv=d_sums.copy_to_host(sums)
            print(vv)
    #t_start3 = timeit.default_timer()
    #n=0
#    for i in range(k_num):
#        #cuda.profile_start()
#        #t_start3 = timeit.default_timer()
#        s=timer()
#        
#        
#        vec_sum_row[len(arr[i].getMat()), threadsperblock](d_rvecs,d_sums)
#        n=n+1
#        #cuda.synchronize()
#        print((timer()-s)*1000000)
#        print("==========")
#        t=timer()
#        vv=d_sums.copy_to_host(sums)
#        print(vv)
        
testProfile()
