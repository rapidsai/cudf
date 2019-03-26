#include "rolling.hpp"

// custom aggregator
__host__ __device__
int custom_agg(const int *data, const int window)
{
  int res = 0;
  for (int i = 0; i < window; i++)
    res += data[i];
  return res;
}
__device__ custom_agg_funtype gpu_custom_agg = custom_agg;

int main(int argc, char **argv)
{
  int size = 10;
  int window = 2;

  int *input;
  int *output;
  cudaMallocManaged(&input, size * sizeof(int));
  cudaMallocManaged(&output, size * sizeof(int));
  cudaMemset(output, 0, size * sizeof(int));

  // initialize with random numbers
  for(int i = 0; i < size; i++) {
    input[i] = std::rand() % 10;
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;

  // copy function pointer to the cpu
  custom_agg_funtype host_custom_agg;
  cudaMemcpyFromSymbol(&host_custom_agg, gpu_custom_agg, sizeof(custom_agg_funtype));
 
  // call cudf
  cudf_rolling(output, input, size, window, host_custom_agg);

  // compute cpu result
  for(int i = 0; i < size; i++) {
    std::cout << output[i] << " ";
    assert(output[i] == custom_agg(input + i, window));
  }
  std::cout << std::endl;

  return 0;
}
