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
// The replace_re function below will use this to allocate the memory needed for the result.
// The operation phase will do the actual replace into the new memory.
// Combining the two phases into a single function minimizes the build time
// due to inlining the regex calls.
template <size_t stack_size>
struct replace_multi_regex_fn {
  custring_view_array d_strings;
  dreprog** d_progs;
  unsigned int progs_count;
  custring_view_array d_repls;
  unsigned int repl_count;
  size_t* d_offsets;
  bool bcompute_size_only{true};  // minimizes build time without introducing divergence
  char* d_buffer;
  custring_view_array d_results;
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    u_char data1[stack_size], data2[stack_size];
    unsigned int nchars = dstr->chars_count();  // number of characters in input string
    unsigned int nbytes = dstr->size();         // number of bytes in input string
    char* buffer        = nullptr;              // output buffer
    char* sptr          = dstr->data();         // input buffer
    if (!bcompute_size_only) buffer = d_buffer + d_offsets[idx];
    char* optr        = buffer;  // running output pointer
    unsigned int size = nchars, spos = 0, lpos = 0;
    // walk through string looking for matches at each character
    while (spos < size) {
      for (unsigned tidx = 0; tidx < progs_count; ++tidx) {
        dreprog* prog = d_progs[tidx];
        prog->set_stack_mem(data1, data2);
        int begin = spos, end = spos + 1;           // look for match only at this pos
        if (prog->find(idx, dstr, begin, end) > 0)  // minimize calls to this
        {                                           // got one, measure or replace
          custring_view* d_repl = (repl_count == 1 ? d_repls[0] : d_repls[tidx]);
          if (bcompute_size_only)  // measure
          {
            nbytes += (d_repl ? d_repl->size() : 0) -
                      (dstr->byte_offset_for(end) - dstr->byte_offset_for(begin));
            nchars += (d_repl ? d_repl->chars_count() : 0) - (end - begin);
          } else                                                    // replace
          {                                                         // i:bbbbsssseeee
            int spos = dstr->byte_offset_for(begin);                //       ^
            copy_and_incr(optr, sptr + lpos, spos - lpos);          // o:bbbb
            if (d_repl)                                             //       ^
              copy_and_incr(optr, d_repl->data(), d_repl->size());  // o:bbbbrrrr
            lpos = dstr->byte_offset_for(end);                      // i:bbbbsssseeee
          }                                                         //           ^
          spos = end - 1;
          break;  // next position
        }
      }
      ++spos;
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

//
NVStrings* NVStrings::replace_re(std::vector<const char*>& patterns, NVStrings& repls)
{
  if (patterns.size() == 0 || repls.size() == 0)
    throw std::invalid_argument("replace_re patterns and repls parameters cannot be empty");
  if (repls.size() > 1 && (repls.size() != patterns.size()))
    throw std::invalid_argument(
      "replace_re patterns and repls must have the same number of strings");
  unsigned int count = size();
  if (count == 0) return new NVStrings(count);
  auto execpol = rmm::exec_policy(0);
  // compile regex's into device objects
  rmm::device_vector<dreprog*> progs;
  int regex_insts = 0;
  for (int idx = 0; idx < (int)patterns.size(); ++idx) {
    const char* pattern = patterns[idx];
    if (!pattern) continue;
    const char32_t* ptn32 = to_char32(pattern);
    dreprog* prog         = dreprog::create_from(ptn32, get_unicode_flags());
    delete ptn32;
    // allocate regex working memory if necessary
    int insts = prog->inst_counts();
    if (insts > MAX_STACK_INSTS) {
      if (!prog->alloc_relists(count)) {
        std::ostringstream message;
        message << "replace_multi_re: number of instructions " << prog->inst_counts();
        message << " and number of strings " << count;
        message << " exceeds available memory";
        dreprog::destroy(prog);
        throw std::invalid_argument(message.str());
      }
    }
    if (insts > regex_insts) regex_insts = insts;
    progs.push_back(prog);
  }
  unsigned int progs_count = (unsigned int)progs.size();
  if (progs_count == 0) throw std::invalid_argument("replace_re invalid patterns");
  dreprog** d_progs           = progs.data().get();
  custring_view_array d_repls = repls.pImpl->getStringsPtr();
  unsigned int repl_count     = repls.size();

  // compute size of the output
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<size_t> offsets(count, 0);
  size_t* d_offsets = offsets.data().get();

  NVStrings* rtn                = nullptr;
  char* d_buffer                = nullptr;
  custring_view_array d_results = nullptr;

  // first loop will compute size output
  // 2nd loop will do the operation in the allocated memory
  enum scan_and_operate { scan, operate };
  auto op = scan;
  while (true) {
    if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
      thrust::for_each_n(execpol->on(0),
                         thrust::make_counting_iterator<unsigned int>(0),
                         count,
                         replace_multi_regex_fn<RX_STACK_SMALL>{d_strings,
                                                                d_progs,
                                                                progs_count,
                                                                d_repls,
                                                                repl_count,
                                                                d_offsets,
                                                                (op == scan),
                                                                d_buffer,
                                                                d_results});
    else if (regex_insts <= 100)
      thrust::for_each_n(execpol->on(0),
                         thrust::make_counting_iterator<unsigned int>(0),
                         count,
                         replace_multi_regex_fn<RX_STACK_MEDIUM>{d_strings,
                                                                 d_progs,
                                                                 progs_count,
                                                                 d_repls,
                                                                 repl_count,
                                                                 d_offsets,
                                                                 (op == scan),
                                                                 d_buffer,
                                                                 d_results});
    else
      thrust::for_each_n(execpol->on(0),
                         thrust::make_counting_iterator<unsigned int>(0),
                         count,
                         replace_multi_regex_fn<RX_STACK_LARGE>{d_strings,
                                                                d_progs,
                                                                progs_count,
                                                                d_repls,
                                                                repl_count,
                                                                d_offsets,
                                                                (op == scan),
                                                                d_buffer,
                                                                d_results});
    // loop diverter
    if (op == operate) break;
    op = operate;
    // allocate output
    rtn      = new NVStrings(count);
    d_buffer = rtn->pImpl->createMemoryFor(d_offsets);
    if (!d_buffer) break;
    // create offsets
    thrust::exclusive_scan(execpol->on(0), offsets.begin(), offsets.end(), offsets.begin());
    d_results = rtn->pImpl->getStringsPtr();
  }
  // cleanup
  for (auto itr = progs.begin(); itr != progs.end(); itr++) dreprog::destroy(*itr);
  return rtn;
}
