/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <exception>
#include <sstream>

#include "nvstrings/NVStrings.h"

#include "../custring_view.cuh"
#include "../regex/regex.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"
#include "./NVStringsImpl.h"

//
// This functor is used with the extract_record method in two ways.
// First, it computes the output size of each string.
// The extract_record needs this in order allocate the memory required
// for the output -- new instance per string.
// The 2nd call does the actual extract into new memory provided.
// Combining the two into a single functor doubles the speed of the build
// since inlining of the regex code is minimized.
// There should be no divergence since all kernels pass through either
// the compute-size branch or not -- there is no mixture within the same
// kernel launch.
//
template <size_t stack_size>
struct extract_record_fn {
  dreprog* prog;
  custring_view_array d_strings;
  int groups;
  int* d_lengths;
  bool bcompute_size_only{true};
  char** d_buffers;
  custring_view_array* d_rows;
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);
    int begin = 0, end = dstr->chars_count();
    if (prog->find(idx, dstr, begin, end) <= 0) return;
    int* sizes                = d_lengths + (idx * groups);
    char* buffer              = nullptr;
    custring_view_array d_row = nullptr;
    if (!bcompute_size_only) {
      buffer = (char*)d_buffers[idx];
      d_row  = d_rows[idx];
    }
    for (int col = 0; col < groups; ++col) {
      int spos = begin, epos = end;
      if (prog->extract(idx, dstr, spos, epos, col) <= 0) continue;
      if (bcompute_size_only) {
        unsigned int size = dstr->substr_size(spos, epos - spos);
        sizes[col]        = (size_t)ALIGN_SIZE(size);
      } else {
        d_row[col] = dstr->substr((unsigned)spos, (unsigned)(epos - spos), 1, buffer);
        buffer += sizes[col];
      }
    }
  }
};

//
// Extract strings into new instance per string as specified and found by the given regex pattern.
//
int NVStrings::extract_record(const char* pattern, std::vector<NVStrings*>& results)
{
  if (pattern == 0) return -1;
  unsigned int count = size();
  if (count == 0) return 0;

  auto execpol = rmm::exec_policy(0);
  // compile regex into device object
  const char32_t* ptn32 = to_char32(pattern);
  dreprog* prog         = dreprog::create_from(ptn32, get_unicode_flags());
  delete ptn32;
  // allocate regex working memory if necessary
  int regex_insts = prog->inst_counts();
  if (regex_insts > MAX_STACK_INSTS) {
    if (!prog->alloc_relists(count)) {
      std::ostringstream message;
      message << "nvstrings::extract_record: number of instructions (" << prog->inst_counts()
              << ") ";
      message << "and number of strings (" << count << ") ";
      message << "exceeds available memory";
      dreprog::destroy(prog);
      throw std::invalid_argument(message.str());
    }
  }
  //
  int groups = prog->group_counts();
  if (groups == 0) {
    dreprog::destroy(prog);
    return 0;
  }
  // compute lengths of each group for each string
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<int> lengths(count * groups, 0);
  int* d_lengths = lengths.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_SMALL>{prog, d_strings, groups, d_lengths});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_MEDIUM>{prog, d_strings, groups, d_lengths});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_LARGE>{prog, d_strings, groups, d_lengths});
  //
  CUDA_TRY(cudaDeviceSynchronize());
  // this part will be slow for large number of strings
  rmm::device_vector<custring_view_array> strings(count, nullptr);
  rmm::device_vector<char*> buffers(count, nullptr);
  for (unsigned int idx = 0; idx < count; ++idx) {
    NVStrings* row = new NVStrings(groups);
    results.push_back(row);
    int* sizes = d_lengths + (idx * groups);
    int size   = thrust::reduce(execpol->on(0), sizes, sizes + groups);
    if (size == 0) continue;
    char* d_buffer = device_alloc<char>(size, 0);
    row->pImpl->setMemoryBuffer(d_buffer, size);
    strings[idx] = row->pImpl->getStringsPtr();
    buffers[idx] = d_buffer;
  }
  // copy each subgroup into each rows memory
  custring_view_array* d_rows = strings.data().get();
  char** d_buffers            = buffers.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_SMALL>{
                         prog, d_strings, groups, d_lengths, false, d_buffers, d_rows});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_MEDIUM>{
                         prog, d_strings, groups, d_lengths, false, d_buffers, d_rows});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       extract_record_fn<RX_STACK_LARGE>{
                         prog, d_strings, groups, d_lengths, false, d_buffers, d_rows});
  //
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "nvs-extract_record(%s): groups=%d\n", pattern, groups);
    printCudaError(err);
  }
  dreprog::destroy(prog);
  return groups;
}
