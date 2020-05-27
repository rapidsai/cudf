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
#include <exception>
#include <sstream>

#include "nvstrings/NVStrings.h"

#include "../custring_view.cuh"
#include "../regex/regex.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"
#include "./NVStringsImpl.h"

// This functor is used to record the substring positions for each
// extract column. Then, only substr is needed for the result.
template <size_t stack_size>
struct extract_sizer_fn {
  dreprog* prog;
  custring_view_array d_strings;
  int col;
  int* d_begins;
  int* d_ends;
  size_t* d_lengths;
  __device__ void operator()(unsigned int idx)
  {
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);
    custring_view* dstr = d_strings[idx];
    d_begins[idx]       = -1;
    d_ends[idx]         = -1;
    if (!dstr) return;
    int begin = 0, end = dstr->chars_count();
    int result = prog->find(idx, dstr, begin, end);
    if (result > 0) result = prog->extract(idx, dstr, begin, end, col);
    if (result > 0) {
      d_begins[idx]     = begin;
      d_ends[idx]       = end;
      unsigned int size = dstr->substr_size(begin, end - begin);
      d_lengths[idx]    = (size_t)ALIGN_SIZE(size);
    }
  }
};

// column-major version of extract() method above
int NVStrings::extract(const char* pattern, std::vector<NVStrings*>& results)
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
  if (regex_insts > MAX_STACK_INSTS)

  {
    if (!prog->alloc_relists(count)) {
      std::ostringstream message;
      message << "nvstrings::extract: number of instructions (" << prog->inst_counts() << ") ";
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
  //
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<int> begins(count, 0);
  int* d_begins = begins.data().get();
  rmm::device_vector<int> ends(count, 0);
  int* d_ends = ends.data().get();
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  // build strings vector for each group (column)
  for (int col = 0; col < groups; ++col) {
    // first, build two vectors of (begin,end) position values;
    // also get the lengths of the substrings
    if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        extract_sizer_fn<RX_STACK_SMALL>{prog, d_strings, col, d_begins, d_ends, d_lengths});
    else if (regex_insts <= 100)
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        extract_sizer_fn<RX_STACK_MEDIUM>{prog, d_strings, col, d_begins, d_ends, d_lengths});
    else
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        extract_sizer_fn<RX_STACK_LARGE>{prog, d_strings, col, d_begins, d_ends, d_lengths});
    // create list of strings for this group
    NVStrings* column = new NVStrings(count);
    results.push_back(column);  // append here so continue statement will work
    char* d_buffer = column->pImpl->createMemoryFor(d_lengths);
    if (d_buffer == 0) continue;
    rmm::device_vector<size_t> offsets(count, 0);
    thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
    // copy the substrings into the new object
    custring_view_array d_results = column->pImpl->getStringsPtr();
    size_t* d_offsets             = offsets.data().get();
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      [d_strings, d_begins, d_ends, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
        custring_view* dstr = d_strings[idx];
        if (!dstr) return;
        int start = d_begins[idx];
        int stop  = d_ends[idx];
        if (stop > start)
          d_results[idx] =
            dstr->substr((unsigned)start, (unsigned)(stop - start), 1, d_buffer + d_offsets[idx]);
      });
    // column already added to results above
  }
  dreprog::destroy(prog);
  return groups;
}
