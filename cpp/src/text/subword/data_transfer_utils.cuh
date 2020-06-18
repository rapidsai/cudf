#pragma once

#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <rmm/thrust_rmm_allocator.h>

inline void gpuCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
      std::cerr << cudaGetErrorString(err) << " in file " << file << " at line "
                                           << line << "." << std::endl;
      exit(1);
  }
}

#define assertCudaSuccess(cu_err) {gpuCheck((cu_err), __FILE__, __LINE__);}