import cudf as gd
import numpy as np
from numba import cuda,jit,float32
import math
TPB = 32 # threads per block, multiples of 32 in general
TPB1 = TPB+1 # trick to reuse shared memory
@cuda.jit(device=True) 
def initialize(array,value,N):
    # N<=len(array)
    for i in range(cuda.threadIdx.x, N, cuda.blockDim.x):
        array[i] = value

@cuda.jit(device=True)
def reduction_sum_SM(array):
    # array is in shared memory
    # len(array) == TPB 
    # the final result is in array[0]
    tid = cuda.threadIdx.x
    j = TPB//2 #16
    while j>0:
       if tid<j:
           array[tid] += array[tid+j]
       j = j//2
       cuda.syncthreads()

@cuda.jit(device=True)            
def compute_mean(array,mean): 
    # mean is a shared memory array
    # the kernel has only one TB
    # the final result is in mean[0]
    tid = cuda.threadIdx.x
    initialize(mean,0,TPB)
    cuda.syncthreads()
   
    tid = cuda.threadIdx.x 
    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        mean[tid] += array[i]
    cuda.syncthreads()

    reduction_sum_SM(mean)
    if tid == 0: 
        mean[0]/=len(array)
    
@cuda.jit(device=True)
def compute_std_with_mean(array,std,mean):
    # std is a shared memory array
    # mean is a scaler, the mean value of array
    # len(std) == TPB
    # the kernel has only one TB
    # the final result is in std[0]
    tid = cuda.threadIdx.x
    initialize(std,0,len(std))
    cuda.syncthreads()

    tid = cuda.threadIdx.x
    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        std[tid] += (array[i]-mean)**2
    cuda.syncthreads()

    reduction_sum_SM(std)
    if tid == 0:
        std[0] = math.sqrt(std[0]/(len(array)-1))
    cuda.syncthreads()

@cuda.jit(device=True)
def compute_skew_with_mean(array,skew,mean):
    # skew is a shared memory array
    # mean is a scaler, the mean value of array
    # len(skew) == TPB+2
    # the kernel has only one TB
    # the final result is in skew[0]
    tid = cuda.threadIdx.x
    initialize(skew,0,len(skew))
    cuda.syncthreads()

    tid = cuda.threadIdx.x
    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        skew[tid] += (array[i]-mean)**2
    cuda.syncthreads()

    reduction_sum_SM(skew)
    if tid == 0:
        skew[TPB] = skew[0]/(len(array))
    cuda.syncthreads()

    initialize(skew,0,TPB)
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        skew[tid] += (array[i]-mean)**3
    cuda.syncthreads()

    reduction_sum_SM(skew)
    if tid == 0:
        n = len(array)
        m3 = skew[0]/(len(array))
        m2 = skew[TPB]
        if m2>0 and n>2:
            skew[0] = math.sqrt((n-1.0)*n)/(n-2.0)*m3/m2**1.5
        else:
            skew[0] = 0
    cuda.syncthreads()

@cuda.jit(device=True)
def compute_kurtosis_with_mean(array,skew,mean):
    # skew is a shared memory array
    # mean is a scaler, the mean value of array
    # len(skew) == TPB+2
    # the kernel has only one TB
    # the final result is in skew[0]
    tid = cuda.threadIdx.x
    initialize(skew,0,len(skew))
    cuda.syncthreads()

    tid = cuda.threadIdx.x
    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        skew[tid] += (array[i]-mean)**2
    cuda.syncthreads()

    reduction_sum_SM(skew)
    if tid == 0:
        skew[TPB] = skew[0]/(len(array))
    cuda.syncthreads()

    initialize(skew,0,TPB)
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(array), cuda.blockDim.x):
        skew[tid] += (array[i]-mean)**4
    cuda.syncthreads()

    reduction_sum_SM(skew)
    if tid == 0:
        n = len(array)
        m4 = skew[0]/(len(array))
        m2 = skew[TPB]
        #skew[0] = math.sqrt((n-1.0)*n)/(n-2.0)*m3/m2**1.5
        if n>3 and m2>0:
            skew[0] = 1.0/(n-2)/(n-3)*((n*n-1.0)*m4/m2**2.0-3*(n-1)**2.0)
        else:
            skew[0] = 0
    cuda.syncthreads()

@cuda.jit(device=True)
def compute_std(array,std):
    # std is a shared memory array
    # len(std) == TPB+1
    # the kernel has only one TB
    # the final result is in std[0]
    compute_mean(array,std)
    std[TPB] = std[0]
    mean = std[TPB]
    compute_std_with_mean(array,std,mean)

@cuda.jit(device=True)
def compute_skew(array,skew):
    # std is a shared memory array
    # len(std) == TPB+1
    # the kernel has only one TB
    # the final result is in std[0]
    compute_mean(array,skew)
    skew[TPB] = skew[0]
    mean = skew[TPB]
    compute_skew_with_mean(array,skew,mean)

@cuda.jit(device=True)
def compute_kurtosis(array,skew):
    # std is a shared memory array
    # len(std) == TPB+1
    # the kernel has only one TB
    # the final result is in std[0]
    compute_mean(array,skew)
    skew[TPB] = skew[0]
    mean = skew[TPB]
    compute_kurtosis_with_mean(array,skew,mean)

@cuda.jit
def compute_mean_kernel(array,out):
    mean = cuda.shared.array(shape=(TPB), dtype=float32)
    compute_mean(array,mean)
    if cuda.threadIdx.x==0:
        out[0] = mean[0]
    cuda.syncthreads()

@cuda.jit
def compute_std_kernel(array,out):
    std = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_std(array,std)
    if cuda.threadIdx.x==0:
        out[0] = std[0]
    cuda.syncthreads()

@cuda.jit
def compute_skew_kernel(array,out):
    skew = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_skew(array,skew)
    if cuda.threadIdx.x==0:
        out[0] = skew[0]
    cuda.syncthreads()

@cuda.jit
def compute_kurtosis_kernel(array,out):
    skew = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_kurtosis(array,skew)
    if cuda.threadIdx.x==0:
        out[0] = skew[0]
    cuda.syncthreads()

@cuda.jit(device=True)
def gd_group_apply_std(ds_in,ds_out):
    std = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_std(ds_in,std)
    for i in range(cuda.threadIdx.x, len(ds_in), cuda.blockDim.x):
        ds_out[i] = std[0]

@cuda.jit(device=True)
def gd_group_apply_var(ds_in,ds_out):
    std = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_std(ds_in,std)
    for i in range(cuda.threadIdx.x, len(ds_in), cuda.blockDim.x):
        ds_out[i] = std[0]**2

@cuda.jit(device=True)
def gd_group_apply_skew(ds_in,ds_out):
    skew = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_skew(ds_in,skew)
    for i in range(cuda.threadIdx.x, len(ds_in), cuda.blockDim.x):
        ds_out[i] = skew[0]

@cuda.jit(device=True)
def gd_group_apply_kurtosis(ds_in,ds_out):
    kurtosis = cuda.shared.array(shape=(TPB1), dtype=float32)
    compute_kurtosis(ds_in,kurtosis)
    for i in range(cuda.threadIdx.x, len(ds_in), cuda.blockDim.x):
        ds_out[i] = kurtosis[0]

def groupby_median(df,idcol,col):
    outcol = 'median_%s'%(col)
    func = \
    '''def median(df):\n
           df["%s"] = df["%s"].nsmallest((len(df)+1)//2)[-1]
           return df
    '''%(outcol,col)
    exec(func)
    func = eval('median')
    df = df.groupby(idcol, method="cudf").apply(func)
    return df

def cudf_groupby_agg(df,idcol,col,func_name):
    # python trick to get named arguments
    if func_name in ['mean','max','min','sum','count']:
        return df.groupby(idcol).agg({col:func_name})
    outcol = '%s_%s'%(func_name,col)
    if func_name == 'median':
        df = groupby_median(df,idcol,col)
    else:
        fn = '%s(%s,%s)'%(func_name,col,outcol)
        func = \
        '''def %s:\n
               gd_group_apply_%s
        '''%(fn,fn)
        exec(func)
        func = eval(func_name)
        df = df.groupby(idcol,method='cudf').apply_grouped(func,
                                  incols=[col],
                                  outcols={outcol: np.float32},
                                  tpb=TPB)
    dg = df.groupby(idcol).agg({outcol:'mean'})
    df.drop_column(outcol)
    meancol = 'mean_%s'%outcol
    dg[outcol] = dg[meancol]
    dg.drop_column(meancol)
    return dg

def cudf_groupby_aggs(df,group_id_col,aggs):
    """
    Parameters
    ----------
    df : cudf dataframe
        dataframe to be grouped
    group_id_col : string
        name of the column which is used as the key of the group
    aggs : dictionary
        key is the name of column for which aggregation is calculated
        values is the name of function for aggregation
    Returns
    -------
    dg : cudf dataframe
        result of groupby aggregation
    """
    dg = None
    for col,funcs in aggs.items():
        for func in funcs:
            if dg is None:
                dg = cudf_groupby_agg(df,group_id_col,col,func)
            else:
                tmp = cudf_groupby_agg(df,group_id_col,col,func)
                dg = dg.merge(tmp,on=[group_id_col],how='left')
    return dg
