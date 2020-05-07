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
#include <thrust/extrema.h>
#include <thrust/for_each.h>
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
// This column version is less intense than its record counterpart.
template <size_t stack_size>
struct findall_fn {
  dreprog* prog;
  custring_view_array d_strings;
  int* d_counts;
  bool bcompute_size_only{true};
  int column;
  thrust::pair<const char*, size_t>* d_indexes;
  //
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    if (!bcompute_size_only && (column >= d_counts[idx])) return;
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);
    if (!bcompute_size_only) {
      d_indexes[idx].first  = nullptr;  // initialize to
      d_indexes[idx].second = 0;        // null string
    }
    int spos = 0, nchars = (int)dstr->chars_count();
    int epos = nchars, column_count = 0;
    // prog->find(idx,dstr,spos,epos);
    // for( int col=0; col <= column; ++c )
    while (spos <= nchars) {
      if (prog->find(idx, dstr, spos, epos) <= 0) break;
      if (!bcompute_size_only && (column_count == column)) break;
      spos = epos > spos ? epos : spos + 1;
      epos = nchars;
      ++column_count;
      // prog->find(idx,dstr,spos,epos);
    }
    if (bcompute_size_only)
      d_counts[idx] = column_count;
    else {
      // this will be the string for this column
      if (spos < epos) {
        spos                  = dstr->byte_offset_for(spos);  // convert char pos
        epos                  = dstr->byte_offset_for(epos);  // to byte offset
        d_indexes[idx].first  = dstr->data() + spos;
        d_indexes[idx].second = (epos - spos);
      } else {  // create empty string instead of a null one
        d_indexes[idx].first = dstr->data();
      }
    }
  }
};

// same as findall but strings are returned organized in column-major
int NVStrings::findall(const char* pattern, std::vector<NVStrings*>& results)
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
      message << "nvstrings::findall: number of instructions (" << prog->inst_counts() << ") ";
      message << "and number of strings (" << count << ") ";
      message << "exceeds available memory";
      dreprog::destroy(prog);
      throw std::invalid_argument(message.str());
    }
  }

  // compute counts of each match and size of the buffers
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<int> counts(count, 0);
  int* d_counts = counts.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_fn<RX_STACK_SMALL>{prog, d_strings, d_counts});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_fn<RX_STACK_MEDIUM>{prog, d_strings, d_counts});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       findall_fn<RX_STACK_LARGE>{prog, d_strings, d_counts});
  int columns = *thrust::max_element(execpol->on(0), counts.begin(), counts.end());
  // boundary case: if no columns, return one null column (issue #119)
  if (columns == 0) results.push_back(new NVStrings(count));

  // create columns of nvstrings
  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    // build index for each string -- collect pointers and lengths
    rmm::device_vector<thrust::pair<const char*, size_t>> indexes(count);
    thrust::pair<const char*, size_t>* d_indexes = indexes.data().get();
    if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        findall_fn<RX_STACK_SMALL>{prog, d_strings, d_counts, false, col_idx, d_indexes});
    else if (regex_insts <= 100)
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        findall_fn<RX_STACK_MEDIUM>{prog, d_strings, d_counts, false, col_idx, d_indexes});
    else
      thrust::for_each_n(
        execpol->on(0),
        thrust::make_counting_iterator<unsigned int>(0),
        count,
        findall_fn<RX_STACK_LARGE>{prog, d_strings, d_counts, false, col_idx, d_indexes});
    NVStrings* column =
      NVStrings::create_from_index((std::pair<const char*, size_t>*)d_indexes, count);
    results.push_back(column);
  }
  dreprog::destroy(prog);
  return (unsigned int)results.size();
}
