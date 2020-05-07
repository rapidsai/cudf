#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <memory>
#include <string>
#include <vector>

#include "nvstrings/NVCategory.h"
#include "nvstrings/NVStrings.h"

//
// cd ../cpp/build
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70 ../tests/catcopy.cu
// -L. -lNVStrings -lNVCategory -o catcopy --linker-options -rpath,.:
//

const char* dataset[]   = {"aaa", "bbb", "bbb", "fff", "ccc", "fff", 0, "", "z", "y"};
unsigned int dset_count = 10;
// these two values use about 4GB of GPU memory
int num_strings = 1000;
int num_copies  = 100000;

int main(int argc, const char** argv)
{
  const char** hstrs = dataset;
  unsigned int count = dset_count;
  if (argc > 1) {
    hstrs = &argv[1];
    count = (unsigned int)(argc - 1);
  }

  NVStrings* dstrs = NVStrings::create_from_array(hstrs, count);

  std::vector<NVStrings*> strslist;
  for (int idx = 0; idx < num_strings; ++idx) strslist.push_back(dstrs);

  NVCategory* dcat = NVCategory::create_from_strings(strslist);
  printf("number of keys = %u\n", dcat->keys_size());
  printf("number of values = %u\n", dcat->size());
  fflush(0);
  NVStrings::destroy(dstrs);

  printf("creating copies\n");
  std::vector<NVCategory*> cats;
  for (int idx = 0; idx < num_copies; ++idx) {
    NVCategory* dcc = dcat->copy();
    // printf("%d:%p\n",idx,dcc);
    cats.push_back(dcc);
    dcat = dcc;
  }
  printf("press enter to continue\n");
  std::getchar();
  // printf("destroying dcat\n");
  // NVCategory::destroy(dcat);
  printf("destroying copies\n");
  for (int idx = 0; idx < (int)cats.size(); ++idx) {
    NVCategory* dcc = cats[idx];
    // printf("~%d:%p\n",idx,dcc);
    NVCategory::destroy(dcc);
  }

  return 0;
}