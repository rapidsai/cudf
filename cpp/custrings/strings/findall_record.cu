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

// Like the other regex functors, this one has two modes: size/count calculation
// and then the operation itself (findall). This minimizes the inlining of
// the regex code while not causing divergence. Makes the code a bit messy
// but build times are reduced by half since only one regex find() is inlined.
template <size_t stack_size>
struct findall_record_fn {
  dreprog* prog;
  custring_view_array d_strings;
  int* d_counts;
  int* d_sizes;
  bool bcompute_size_only{true};
  char** d_buffers;
  custring_view_array* d_rows;
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);

    if (!bcompute_size_only && (d_counts[idx] < 1)) return;
    char* buffer             = nullptr;
    custring_view_array drow = nullptr;
    if (!bcompute_size_only) {
      buffer = (char*)d_buffers[idx];
      drow   = d_rows[idx];
    }

    int nbytes = 0, nchars = (int)dstr->chars_count();
    int spos = 0, rows_idx = 0, find_count = 0;
    while (spos <= nchars) {
      int epos = nchars;
      if (prog->find(idx, dstr, spos, epos) <= 0) break;
      if (bcompute_size_only) {
        unsigned int bytes = (dstr->byte_offset_for(epos) - dstr->byte_offset_for(spos));
        unsigned int size  = custring_view::alloc_size(bytes, (epos - spos));
        nbytes += ALIGN_SIZE(size);
        ++find_count;
      } else {
        custring_view* str = dstr->substr((unsigned)spos, (unsigned)(epos - spos), 1, buffer);
        drow[rows_idx++]   = str;
        buffer += ALIGN_SIZE(str->alloc_size());
      }
      spos = epos > spos ? epos : spos + 1;
    }
    if (bcompute_size_only) {
      d_sizes[idx]  = nbytes;
      d_counts[idx] = find_count;
    }
  }
};

// for each string, return substring(s) which match specified pattern
int NVStrings::findall_record(const char* pattern, std::vector<NVStrings*>& results)
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
      message << "nvstrings::findall_record: number of instructions (" << prog->inst_counts()
              << ") ";
      message << "and number of strings (" << count << ") ";
      message << "exceeds available memory";
      dreprog::destroy(prog);
      throw std::invalid_argument(message.str());
    }
  }

  // compute counts of each match and size of the buffers
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<int> sizes(count, 0);
  int* d_sizes = sizes.data().get();
  rmm::device_vector<int> counts(count, 0);
  int* d_counts = counts.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_SMALL>{prog, d_strings, d_counts, d_sizes});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_MEDIUM>{prog, d_strings, d_counts, d_sizes});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_LARGE>{prog, d_strings, d_counts, d_sizes});
  CUDA_TRY(cudaDeviceSynchronize());
  //
  // create rows of buffers
  thrust::host_vector<int> hcounts(counts);  // copies counts from device
  thrust::host_vector<custring_view_array> hrows(count, nullptr);
  thrust::host_vector<char*> hbuffers(count, nullptr);
  for (unsigned int idx = 0; idx < count; ++idx) {
    int rcount     = hcounts[idx];
    NVStrings* row = new NVStrings(rcount);
    results.push_back(row);
    if (rcount == 0) continue;
    hrows[idx]     = row->pImpl->getStringsPtr();
    int size       = sizes[idx];
    char* d_buffer = device_alloc<char>(size, 0);
    row->pImpl->setMemoryBuffer(d_buffer, size);
    hbuffers[idx] = d_buffer;
  }
  // copy substrings into buffers
  rmm::device_vector<custring_view_array> rows(hrows);  // copies hrows to device
  custring_view_array* d_rows = rows.data().get();
  rmm::device_vector<char*> buffers(hbuffers);  // copies hbuffers to device
  char** d_buffers = buffers.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_SMALL>{
                         prog, d_strings, d_counts, d_sizes, false, d_buffers, d_rows});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_MEDIUM>{
                         prog, d_strings, d_counts, d_sizes, false, d_buffers, d_rows});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_record_fn<RX_STACK_LARGE>{
                         prog, d_strings, d_counts, d_sizes, false, d_buffers, d_rows});
  //
  printCudaError(cudaDeviceSynchronize(), "nvs-findall_record");
  dreprog::destroy(prog);
  return (int)results.size();
}
