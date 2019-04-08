#include <cudf.h>
#include <cuda_runtime_api.h>

/**
 * @file implementations of cuDF API calls which are mearly forwarded
 * CUDA Runtime API function calls - regarding CUDA errors.
 *
 * @note the code here is actually pure C.
 */

int gdf_cuda_last_error() {
    return cudaGetLastError();
}

const char * gdf_cuda_error_string(int cuda_error) {
    return cudaGetErrorString((cudaError_t)cuda_error);
}

const char * gdf_cuda_error_name(int cuda_error) {
    return cudaGetErrorName((cudaError_t)cuda_error);
}
