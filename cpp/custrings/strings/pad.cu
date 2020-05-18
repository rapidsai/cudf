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

#include "nvstrings/NVStrings.h"

#include "../custring_view.cuh"
#include "./NVStringsImpl.h"

// duplicate and concatenate the string the number of times specified
NVStrings* NVStrings::repeat(unsigned int reps)
{
  unsigned int count            = size();
  custring_view_array d_strings = pImpl->getStringsPtr();

  auto execpol = rmm::exec_policy(0);
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, reps, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       unsigned int bytes  = dstr->size();
                       unsigned int nchars = dstr->chars_count();
                       if (reps > 1) {
                         bytes += (bytes * (reps - 1));
                         nchars += (nchars * (reps - 1));
                       }
                       unsigned int size = custring_view::alloc_size(bytes, nchars);
                       size              = ALIGN_SIZE(size);
                       d_lengths[idx]    = (size_t)size;
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the repeat
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, reps, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      char* buffer        = d_buffer + d_offsets[idx];
      custring_view* dout = custring_view::create_from(buffer, *dstr);
      int count           = (reps > 1 ? reps : 1);
      while (--count > 0) dout->append(*dstr);  // *dout += *dstr; works too
      d_results[idx] = dout;
    });
  //
  // printCudaError(cudaDeviceSynchronize(),"nvs-repeat");
  return rtn;
}

// Add specified padding to each string.
// Side:{'left','right','both'}, default is 'left'.
NVStrings* NVStrings::pad(unsigned int width, padside side, const char* fillchar)
{
  if (side == right)  // pad to the right
    return ljust(width, fillchar);
  if (side == both)  // pad both ends
    return center(width, fillchar);
  // default is pad to the left
  return rjust(width, fillchar);
}

// Pad the end of each string to the minimum width.
NVStrings* NVStrings::ljust(unsigned int width, const char* fillchar)
{
  unsigned int count            = size();
  custring_view_array d_strings = pImpl->getStringsPtr();

  auto execpol = rmm::exec_policy(0);
  if (!fillchar || *fillchar == 0) fillchar = " ";
  Char d_fillchar      = 0;
  unsigned int fcbytes = custring_view::char_to_Char(fillchar, d_fillchar);

  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       unsigned int bytes  = dstr->size();
                       unsigned int nchars = dstr->chars_count();
                       if (width > nchars) {
                         unsigned int pad = width - nchars;
                         bytes += fcbytes * pad;
                         nchars += pad;
                       }
                       unsigned int size = custring_view::alloc_size(bytes, nchars);
                       size              = ALIGN_SIZE(size);
                       d_lengths[idx]    = (size_t)size;
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;  // all strings are null
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the padding
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, width, d_fillchar, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      // create init string with size enough for inserts
      char* buffer        = d_buffer + d_offsets[idx];
      custring_view* dout = custring_view::create_from(buffer, *dstr);
      unsigned int nchars = dstr->chars_count();
      if (width > nchars)  // add pad character to the end
        dout->insert(nchars, width - nchars, d_fillchar);
      d_results[idx] = dout;
    });
  //
  // printCudaError(cudaDeviceSynchronize(),"nvs-ljust");
  return rtn;
}

// Pad the beginning and end of each string to the minimum width.
NVStrings* NVStrings::center(unsigned int width, const char* fillchar)
{
  unsigned int count        = size();
  custring_view** d_strings = pImpl->getStringsPtr();

  auto execpol = rmm::exec_policy(0);
  if (!fillchar || *fillchar == 0) fillchar = " ";
  Char d_fillchar = 0;
  int fcbytes     = custring_view::char_to_Char(fillchar, d_fillchar);

  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       unsigned int bytes  = dstr->size();
                       unsigned int nchars = dstr->chars_count();
                       if (width > nchars) {
                         unsigned int pad = width - nchars;
                         bytes += fcbytes * pad;
                         nchars += pad;
                       }
                       unsigned int size = custring_view::alloc_size(bytes, nchars);
                       size              = ALIGN_SIZE(size);
                       d_lengths[idx]    = (size_t)size;
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the padding
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, width, d_fillchar, fcbytes, d_buffer, d_offsets, d_results] __device__(
      unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      // create init string with buffer sized enough the inserts
      char* buffer        = d_buffer + d_offsets[idx];
      custring_view* dout = custring_view::create_from(buffer, *dstr);
      unsigned int nchars = dstr->chars_count();
      if (width > nchars) {
        unsigned int pad   = width - nchars;
        unsigned int left  = pad / 2;
        unsigned int right = pad - left;
        dout->insert(nchars, right, d_fillchar);
        dout->insert(0, left, d_fillchar);
      }
      d_results[idx] = dout;
    });
  //
  // printCudaError(cudaDeviceSynchronize(),"nvs-center");
  return rtn;
}

// Pad the beginning of each string to the minimum width.
NVStrings* NVStrings::rjust(unsigned int width, const char* fillchar)
{
  unsigned int count        = size();
  custring_view** d_strings = pImpl->getStringsPtr();

  auto execpol = rmm::exec_policy(0);
  if (!fillchar || *fillchar == 0) fillchar = " ";
  Char d_fillchar = 0;
  int fcbytes     = custring_view::char_to_Char(fillchar, d_fillchar);

  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, width, fcbytes, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       unsigned int bytes  = dstr->size();
                       unsigned int nchars = dstr->chars_count();
                       if (width > nchars) {
                         unsigned int pad = width - nchars;
                         bytes += fcbytes * pad;
                         nchars += pad;
                       }
                       unsigned int size = custring_view::alloc_size(bytes, nchars);
                       size              = ALIGN_SIZE(size);
                       d_lengths[idx]    = (size_t)size;
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the padding
  custring_view** d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets         = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<size_t>(0),
    count,
    [d_strings, width, d_fillchar, d_buffer, d_offsets, d_results] __device__(size_t idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      // create init string with size enough for inserts
      char* buffer        = d_buffer + d_offsets[idx];
      custring_view* dout = custring_view::create_from(buffer, *dstr);
      unsigned int nchars = dstr->chars_count();
      if (width > nchars)  // add pad character to the beginning
        dout->insert(0, width - nchars, d_fillchar);
      d_results[idx] = dout;
    });
  //
  // printCudaError(cudaDeviceSynchronize(),"nvs-rjust");
  return rtn;
}

// Pad the beginning of each string with 0s honoring any sign prefix.
NVStrings* NVStrings::zfill(unsigned int width)
{
  unsigned int count        = size();
  custring_view** d_strings = pImpl->getStringsPtr();

  auto execpol = rmm::exec_policy(0);
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, width, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       unsigned int bytes  = dstr->size();
                       unsigned int nchars = dstr->chars_count();
                       if (width > nchars) {
                         unsigned int pad = width - nchars;
                         bytes += pad;
                         nchars += pad;
                       }
                       unsigned int size = custring_view::alloc_size(bytes, nchars);
                       size              = ALIGN_SIZE(size);
                       d_lengths[idx]    = (size_t)size;
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the fill
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<size_t>(0),
                     count,
                     [d_strings, width, d_buffer, d_offsets, d_results] __device__(size_t idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       // create init string with buffer sized enough for the inserts
                       char* buffer        = d_buffer + d_offsets[idx];
                       char* sptr          = dstr->data();
                       unsigned int sz     = dstr->size();
                       custring_view* dout = custring_view::create_from(buffer, sptr, sz);
                       unsigned int nchars = dstr->chars_count();
                       if (width > nchars) {
                         char fchr = ((sz <= 0) ? 0 : *sptr);  // check for sign and shift
                         unsigned int pos =
                           (((fchr == '-') || (fchr == '+')) ? 1 : 0);  // insert pos if necessary
                         dout->insert(pos, width - nchars, '0');        // insert characters
                       }
                       d_results[idx] = dout;
                     });
  //
  // printCudaError(cudaDeviceSynchronize(),"nvs-zfill");
  return rtn;
}

// Essentially inserting new-line chars into appropriate places in the string to ensure that each
// 'line' is no longer than width characters. Along the way, tabs may be expanded (8 spaces) or
// replaced. and long words may be broken up or reside on their own line.
//    expand_tabs = false         (tab = 8 spaces)
//    replace_whitespace = true   (replace with space)
//    drop_whitespace = false     (no spaces after new-line)
//    break_long_words = false
//    break_on_hyphens = false
NVStrings* NVStrings::wrap(unsigned int width)
{
  unsigned int count            = size();
  custring_view_array d_strings = pImpl->getStringsPtr();
  auto execpol                  = rmm::exec_policy(0);

  // need to compute the size of each new string
  rmm::device_vector<size_t> sizes(count, 0);
  size_t* d_sizes = sizes.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_sizes] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (!dstr) return;
                       // replacing space with new-line does not change the size
                       // -- this is oversimplification since 'expand' and 'drop' options would
                       // change the size of the string
                       d_sizes[idx] = ALIGN_SIZE(dstr->alloc_size());
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin());
  // do the wrap logic
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, width, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      char* buffer    = d_buffer + d_offsets[idx];
      char* sptr      = dstr->data();
      unsigned int sz = dstr->size();
      // start by copying whole string into buffer
      char* optr = buffer;
      memcpy(optr, sptr, sz);
      // replace appropriate spaces with new-line
      // - this should be way more complicated with all the permutations of flags
      unsigned int nchars       = dstr->chars_count();
      int charOffsetToLastSpace = -1, byteOffsetToLastSpace = -1, spos = 0, bidx = 0;
      for (unsigned int pos = 0; pos < nchars; ++pos) {
        Char chr = dstr->at(pos);
        if (chr <= ' ') {  // convert all whitespace to space
          optr[bidx]            = ' ';
          byteOffsetToLastSpace = bidx;
          charOffsetToLastSpace = pos;
        }
        if ((pos - spos) >= width) {
          if (byteOffsetToLastSpace >= 0) {
            optr[byteOffsetToLastSpace] = '\n';
            spos                        = charOffsetToLastSpace;
            byteOffsetToLastSpace = charOffsetToLastSpace = -1;
          }
        }
        bidx += (int)custring_view::bytes_in_char(chr);
      }
      d_results[idx] = custring_view::create_from(buffer, buffer, sz);
    });
  //
  // cudaError_t err = cudaDeviceSynchronize();
  // if( err != cudaSuccess )
  //{
  //    fprintf(stderr,"nvs-wrap(%d)\n",width);
  //    printCudaError(err);
  //}
  return rtn;
}
