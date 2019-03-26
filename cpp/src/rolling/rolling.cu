#include <cstdlib>
#include <iostream>
#include <assert.h>

// type definition for the aggregation function
typedef int (*custom_agg_funtype)(const int*, const int);

// gpu kernel
__global__ void rolling_kernel(int *output, const int *input, const int size, const int window, custom_agg_funtype agg)
{
  for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid++) {
    int adj_window = tid + window <= size ? window : size - tid;
    output[tid] = agg(input + tid, adj_window);
  }
}

// prototype cudf api
void cudf_rolling(int *output, const int *input, const int size, const int window, custom_agg_funtype agg)
{
  // run the kernel
  int block = 256;
  int grid = (size + block-1) / block;
  rolling_kernel<<<grid, block>>>(output, input, size, window, agg);
  cudaDeviceSynchronize();
}

