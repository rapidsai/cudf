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

// The device function here has two modes: compute-size and replace.
// The compute-size phase just computes the memory needed for the replace.
// The replace_re function will use this to allocate the memory needed for the result.
// The operation phase will do the actual replace into the new memory.
// Combining the two phases into a single function minimizes the build time
// due to inlining the regex calls.
template <size_t stack_size>
struct replace_regex_fn {
  dreprog* prog;
  custring_view_array d_strings;
  custring_view* d_repl;
  int maxrepl;
  size_t* d_offsets;
  bool bcompute_size_only;  // minimizes build time without introducing divergence
  char* d_buffer;
  custring_view_array d_results;
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);
    int mxn             = maxrepl;
    unsigned int nchars = dstr->chars_count();  // number of characters in input string
    unsigned int nbytes = dstr->size();         // number of bytes in input string
    if (mxn < 0) mxn = (int)nchars;             // max possible replaces for this string
    char* buffer = nullptr;                     // output buffer
    char* sptr   = nullptr;                     // input buffer
    char* optr   = nullptr;                     // running output pointer
    if (!bcompute_size_only) {
      buffer = d_buffer + d_offsets[idx];  // output buffer
      sptr   = dstr->data();               // input buffer
      optr   = buffer;                     // running output pointer
    }
    int lpos = 0, begin = 0, end = (int)nchars;  // working vars
    // copy input to output replacing strings as we go
    while (mxn > 0)  // while((result > 0) && (mxn > 0))
    {
      if (prog->find(idx, dstr, begin, end) <= 0) break;
      if (bcompute_size_only) {
        nbytes += d_repl->size() - (dstr->byte_offset_for(end) - dstr->byte_offset_for(begin));
        nchars += d_repl->chars_count() - (end - begin);
      } else {                                                // i:bbbbsssseeee
        int spos = dstr->byte_offset_for(begin);              //       ^
        copy_and_incr(optr, sptr + lpos, spos - lpos);        // o:bbbb
                                                              //       ^
        copy_and_incr(optr, d_repl->data(), d_repl->size());  // o:bbbbrrrr
                                                              //           ^
        lpos = dstr->byte_offset_for(end);                    // i:bbbbsssseeee
      }                                                       //           ^
      begin = end;
      end   = (int)dstr->chars_count();
      --mxn;
    }
    if (bcompute_size_only) {
      unsigned int size = custring_view::alloc_size(nbytes, nchars);
      d_offsets[idx]    = ALIGN_SIZE(size);
    } else {                                     // copy the rest:
      memcpy(optr, sptr + lpos, nbytes - lpos);  // o:bbbbrrrreeee
      unsigned int nsz = (unsigned int)(optr - buffer) + nbytes - lpos;
      d_results[idx]   = custring_view::create_from(buffer, buffer, nsz);
    }
  }
};

// same as above except parameter is regex
NVStrings* NVStrings::replace_re(const char* pattern, const char* repl, int maxrepl)
{
  if (!pattern || !*pattern)
    throw std::invalid_argument("nvstrings::replace_re parameter cannot be null or empty");
  unsigned int count = size();
  if (count == 0) return new NVStrings(count);
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
      message << "nvstrings::replace_re: number of instructions " << prog->inst_counts();
      message << " and number of strings " << count;
      message << " exceeds available memory";
      dreprog::destroy(prog);
      throw std::invalid_argument(message.str());
    }
  }

  //
  // copy replace string to device memory
  if (!repl) repl = "";
  unsigned int repl_length = (unsigned int)strlen(repl);
  unsigned int repl_size   = custring_view::alloc_size(repl, repl_length);
  custring_view* d_repl    = reinterpret_cast<custring_view*>(device_alloc<char>(repl_size, 0));
  custring_view::create_from_host(d_repl, repl, repl_length);

  // compute size of the output
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<size_t> sizes(count, 0);
  size_t* d_sizes = sizes.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      replace_regex_fn<RX_STACK_SMALL>{prog, d_strings, d_repl, maxrepl, d_sizes, true});
  else if (regex_insts <= 100)
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      replace_regex_fn<RX_STACK_MEDIUM>{prog, d_strings, d_repl, maxrepl, d_sizes, true});
  else
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      replace_regex_fn<RX_STACK_LARGE>{prog, d_strings, d_repl, maxrepl, d_sizes, true});

  //
  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
  if (d_buffer == 0) {
    dreprog::destroy(prog);
    RMM_FREE(d_repl, 0);
    return rtn;  // all strings are null
  }
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin());
  // do the replace
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       replace_regex_fn<RX_STACK_SMALL>{
                         prog, d_strings, d_repl, maxrepl, d_offsets, false, d_buffer, d_results});
  else if (regex_insts <= 100)
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       replace_regex_fn<RX_STACK_MEDIUM>{
                         prog, d_strings, d_repl, maxrepl, d_offsets, false, d_buffer, d_results});
  else
    thrust::for_each_n(execpol->on(0),
                       thrust::make_counting_iterator<unsigned int>(0),
                       count,
                       replace_regex_fn<RX_STACK_LARGE>{
                         prog, d_strings, d_repl, maxrepl, d_offsets, false, d_buffer, d_results});
  //
  dreprog::destroy(prog);
  RMM_FREE(d_repl, 0);
  return rtn;
}
