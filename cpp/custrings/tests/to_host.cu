#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>
#include <vector>

#include "nvstrings/NVStrings.h"

//
// cd ../build
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 ../tests/to_host.cu
// -L. -lNVStrings -o to_host --linker-options -rpath,.:
//

const char* hstrs[] = {"John Smith", "Joe Blow", "Jane Smith", nullptr, ""};
int count           = 5;

int main(int argc, char** argv)
{
  NVStrings* strs = NVStrings::create_from_array(hstrs, count);

  thrust::device_vector<int> bytes(count);
  strs->byte_count(bytes.data().get());

  thrust::host_vector<char*> results(count, nullptr);
  for (int idx = 0; idx < count; ++idx) {
    int length = bytes[idx];
    if (length < 0) continue;
    char* str    = new char[length + 1];
    str[length]  = 0;
    results[idx] = str;
  }

  thrust::host_vector<char*> check(results);
  strs->to_host(results.data(), 0, count);

  for (int idx = 0; idx < count; ++idx) {
    char* str = results[idx];
    std::cout << idx << ": ";
    int length = bytes[idx];
    if (length < 0)
      std::cout << "<null>";
    else if (length == 0)
      std::cout << "<empty>";
    else
      std::cout << str;
    if (str != check[idx])
      std::cout << " -- invalid pointer!";
    else
      delete str;
    std::cout << "\n";
  }

  return 0;
}
