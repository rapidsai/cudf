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
#include "../regex/backref.h"
#include "../regex/regex.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"
#include "./NVStringsImpl.h"

// using stack memory is more efficient but we want to keep the size to a minimum
// so we have a small, medium, and large cases handled here
template <size_t stack_size>
struct backrefs_fn {
  dreprog* prog;
  custring_view_array d_strings;
  custring_view* d_repl;
  thrust::pair<int, int>* d_brefs;
  unsigned int refcount;
  size_t* d_offsets;  // store sizes here in size-only mode
  bool bcompute_size_only{true};
  char* d_buffer;
  custring_view_array d_results;
  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr)  // abcd-efgh   X\1+\2Z
      return;  // nulls create nulls                             // ([a-z])-([a-z]) ==>  abcXd+eZfgh
    u_char data1[stack_size], data2[stack_size];
    prog->set_stack_mem(data1, data2);
    char* buffer = nullptr;
    if (!bcompute_size_only) buffer = d_buffer + d_offsets[idx];  // output buffer
    char* optr    = buffer;                                       // running output pointer
    char* drepl   = d_repl->data();                               // fixed input pointer
    int repl_size = (int)d_repl->size();
    char* sptr    = dstr->data();                                       // abcd-efgh
    int nchars = (int)dstr->chars_count(), nbytes = (int)dstr->size();  // ^
    int lpos = 0, begin = 0, end = (int)nchars;
    // insert extracted strings left-to-right
    while (prog->find(idx, dstr, begin, end) > 0)  // this sets up begin/end
    {
      // we have found the section that needs to be replaced
      if (bcompute_size_only) {
        nchars += d_repl->chars_count() - (end - begin);
        nbytes += d_repl->size() - (dstr->byte_offset_for(end) - dstr->byte_offset_for(begin));
      } else
        copy_and_incr(optr, sptr, dstr->byte_offset_for(begin) - lpos);  // abc________
      int ilpos  = 0;      // last end pos of replace template               //    ^
      char* rptr = drepl;  // running ptr for replace template          // X+Z
      for (unsigned int j = 0; j < refcount; ++j)  // eval each ref      // 1st loop      2nd loop
      {                                            // ------------  --------------
        int refidx = d_brefs[j].first;  // backref number             // X+Z           X+Z
        if (!bcompute_size_only)        //  ^              ^
        {
          int len = d_brefs[j].second - ilpos;  // in bytes to copy
          copy_and_incr_both(optr, rptr, len);  // abcX_______   abcXd+_______
          ilpos += len;                         // update last-position
        }
        int spos = begin, epos = end;  // these are modified by extract
        if ((prog->extract(idx, dstr, spos, epos, refidx - 1) <= 0) ||  // d             e
            (epos <= spos))
          continue;                               // no value for this ref
        nchars += epos - spos;                    // add up chars
        spos      = dstr->byte_offset_for(spos);  // convert to bytes
        int bytes = dstr->byte_offset_for(epos) - spos;
        nbytes += bytes;  // add up bytes
        if (!bcompute_size_only)
          copy_and_incr(optr, dstr->data() + spos, bytes);  // abcXd______   abcXd+e______
      }
      if (!bcompute_size_only) {
        if (rptr < drepl + repl_size)  // copy remainder of template // abcXd+eZ___
          copy_and_incr(optr, rptr, (unsigned int)(drepl - rptr) + repl_size);
        lpos = dstr->byte_offset_for(end);
        sptr = dstr->data() + lpos;  // abcd-efgh
      }                              //       ^
      begin = end;
      end   = (int)dstr->chars_count();
    }
    if (bcompute_size_only) {
      unsigned int size = custring_view::alloc_size(nbytes, nchars);
      d_offsets[idx]    = ALIGN_SIZE(size);  // new size for this string
    } else {
      if (sptr < dstr->data() + dstr->size())  // abcXd+eZfgh
        copy_and_incr(optr, sptr, (unsigned int)(dstr->data() - sptr) + dstr->size());
      d_results[idx] =
        custring_view::create_from(buffer, buffer, (unsigned int)(optr - buffer));  // new string
    }
  }
};

// not even close to the others
NVStrings* NVStrings::replace_with_backrefs(const char* pattern, const char* repl)
{
  if (!pattern || !*pattern)
    throw std::invalid_argument(
      "nvstrings::replace_with_backrefs parameter cannot be null or empty");
  unsigned int count = size();
  if (count == 0 || repl == 0) return new NVStrings(count);  // returns all nulls
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
      message << "nvstrings::replace_with_backrefs: number of instructions (" << prog->inst_counts()
              << ") ";
      message << "and number of strings (" << count << ") ";
      message << "exceeds available memory";
      dreprog::destroy(prog);
      throw std::invalid_argument(message.str());
    }
  }
  //
  // parse the repl string for backref indicators
  std::vector<thrust::pair<int, int>> brefs;
  std::string srepl = parse_backrefs(repl, brefs);

  unsigned int repl_length = (unsigned int)srepl.size();
  unsigned int repl_size   = custring_view::alloc_size(srepl.c_str(), repl_length);
  custring_view* d_repl    = reinterpret_cast<custring_view*>(device_alloc<char>(repl_size, 0));
  custring_view::create_from_host(d_repl, srepl.c_str(), repl_length);

  rmm::device_vector<thrust::pair<int, int>> dbrefs(brefs);
  auto d_brefs          = dbrefs.data().get();
  unsigned int refcount = (unsigned int)dbrefs.size();
  // if refcount != prog->group_counts() -- probably should throw exception

  // compute size of the output
  custring_view_array d_strings = pImpl->getStringsPtr();
  rmm::device_vector<size_t> sizes(count, 0);
  size_t* d_sizes = sizes.data().get();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= 10))
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_SMALL>{prog, d_strings, d_repl, d_brefs, refcount, d_sizes});
  else if (regex_insts <= 100)
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_MEDIUM>{prog, d_strings, d_repl, d_brefs, refcount, d_sizes});
  else
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_LARGE>{prog, d_strings, d_repl, d_brefs, refcount, d_sizes});

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
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_SMALL>{
        prog, d_strings, d_repl, d_brefs, refcount, d_offsets, false, d_buffer, d_results});
  else if (regex_insts <= 100)
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_MEDIUM>{
        prog, d_strings, d_repl, d_brefs, refcount, d_offsets, false, d_buffer, d_results});
  else
    thrust::for_each_n(
      execpol->on(0),
      thrust::make_counting_iterator<unsigned int>(0),
      count,
      backrefs_fn<RX_STACK_LARGE>{
        prog, d_strings, d_repl, d_brefs, refcount, d_offsets, false, d_buffer, d_results});

  //
  dreprog::destroy(prog);
  RMM_FREE(d_repl, 0);
  return rtn;
}
