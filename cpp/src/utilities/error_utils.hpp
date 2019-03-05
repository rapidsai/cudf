#ifndef GDF_ERRORUTILS_H
#define GDF_ERRORUTILS_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_TRY( call ) 									                            \
{                                                                     \
    cudaError_t cudaStatus = call;                                    \
    if ( cudaSuccess != cudaStatus )                                  \
    {                                                                 \
        std::cerr << "ERROR: CUDA Runtime call " << #call             \
                  << " in line " << __LINE__                            \
                  << " of file " << __FILE__                            \
                  << " failed with " << cudaGetErrorString(cudaStatus)  \
                  << " (" << cudaStatus << ").\n";                     \
        return GDF_CUDA_ERROR;                          							\
    }												                                          \
}                                                                                                  

#define RMM_TRY(x)  if ((x)!=RMM_SUCCESS) return GDF_MEMORYMANAGER_ERROR;

#define RMM_TRY_CUDAERROR(x)  if ((x)!=RMM_SUCCESS) return cudaPeekAtLastError();

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())


#define GDF_REQUIRE(F, S) if (!(F)) return (S);

#endif // GDF_ERRORUTILS_H
