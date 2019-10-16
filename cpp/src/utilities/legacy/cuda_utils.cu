#include <cudf/cudf.h>


int gdf_cuda_last_error() {
    return cudaGetLastError();
}

const char * gdf_cuda_error_string(int cuda_error) {
    return cudaGetErrorString((cudaError_t)cuda_error);
}

const char * gdf_cuda_error_name(int cuda_error) {
    return cudaGetErrorName((cudaError_t)cuda_error);
}
