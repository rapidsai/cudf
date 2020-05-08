#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <vector>

#include "nvstrings/NVStrings.h"

//
// cd ../build
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 ../tests/example.cu
// -L. -lNVStrings -o example --linker-options -rpath,.:
//

void my_function(NVStrings& strs)
{
  unsigned int count = strs.size();
  // get a list of pointers/sizes
  thrust::device_vector<std::pair<const char*, size_t>> strings(count);
  std::pair<const char*, size_t>* d_strings = strings.data().get();
  strs.create_index(d_strings, count);

  // launch kernel to do something with them
  thrust::device_vector<int> results(count, 0);
  int* d_results = results.data().get();
  thrust::for_each_n(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_results] __device__(unsigned int idx) {
                       std::pair<const char*, size_t> dstr = d_strings[idx];
                       // do something here
                       // dstr.first is pointer to character array
                       // dstr.second is the length of the character array in bytes
                       printf("%d\n", (int)dstr.second);
                     });
}

NVStrings* first_word1(NVStrings& strs)
{
  unsigned int count = strs.size();
  thrust::device_vector<int> indexes(count, -1);
  int* d_indexes = indexes.data().get();
  strs.find(" ", 0, -1, d_indexes);
  return strs.slice_from(0, d_indexes);
}

NVStrings* first_word2(NVStrings& strs)
{
  std::vector<NVStrings*> strings;
  strs.split(" ", 2, strings);
  NVStrings::destroy(strings[1]);
  return strings[0];
}

int* indexes_that_match_this_string(NVStrings& strs, const char* str)
{
  unsigned int count = strs.size();
  thrust::device_vector<int> indexes(count, -1);
  int* d_indexes = indexes.data().get();
  strs.compare(str, d_indexes);
  int matches = thrust::count_if(
    thrust::device, d_indexes, d_indexes + count, [] __device__(int x) { return x == 0; });
  int* rtn = 0;
  cudaMalloc(&rtn, matches * sizeof(int));
  thrust::counting_iterator<int> itr(0);
  thrust::copy_if(thrust::device, itr, itr + count, rtn, [d_indexes] __device__(int i) {
    return d_indexes[i] == 0;
  });
  return rtn;
}

int main(int argc, char** argv)
{
  const char* hstrs[] = {"John Smith", "Joe Blow", "Jane Smith"};
  NVStrings* strs     = NVStrings::create_from_array(hstrs, 3);

  NVStrings* first = first_word1(*strs);

  int* indexes = indexes_that_match_this_string(*first, "Jane");
  int hindexes[3];
  cudaMemcpy(hindexes, indexes, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", hindexes[0]);

  return 0;
}
