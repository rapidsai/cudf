#ifndef GDF_ERRORUTILS_H
#define GDF_ERRORUTILS_H

#define CUDA_TRY( call ) 									                                                         \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
    {                                                                                              \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
        return GDF_CUDA_ERROR;                          										                       \
    }												                                                                       \
}                                                                                                  

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())


#define GDF_REQUIRE(F, S) if (!(F)) return (S);

#endif // GDF_ERRORUTILS_H
