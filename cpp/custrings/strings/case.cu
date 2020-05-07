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

#include "nvstrings/NVStrings.h"

#include "../custring_view.cuh"
#include "../unicode/is_flags.h"
#include "../util.h"
#include "./NVStringsImpl.h"

//
NVStrings* NVStrings::lower()
{
  unsigned int count = size();
  if (count == 0) return new NVStrings(0);
  auto execpol                  = rmm::exec_policy(0);
  custring_view_array d_strings = pImpl->getStringsPtr();
  unsigned char* d_flags        = get_unicode_flags();
  unsigned short* d_cases       = get_charcases();
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (dstr) {
                         unsigned int bytes = 0;
                         for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
                           Char chr         = *itr;
                           unsigned int chw = custring_view::bytes_in_char(chr);
                           unsigned int uni = u82u(chr);
                           unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                           if (IS_UPPER(flg))
                             chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                           bytes += chw;
                         }
                         d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes, dstr->chars_count()));
                       }
                     });
  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the thing
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (dstr) {
        char* buffer       = d_buffer + d_offsets[idx];
        char* ptr          = buffer;
        unsigned int bytes = 0;
        for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
          Char chr         = *itr;
          unsigned int uni = u82u(chr);
          unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
          if (IS_UPPER(flg)) chr = u2u8(d_cases[uni]);
          unsigned int chw = custring_view::Char_to_char(chr, ptr);
          ptr += chw;
          bytes += chw;
        }
        d_results[idx] = custring_view::create_from(buffer, buffer, bytes);
      }
    });
  //
  return rtn;
}

//
NVStrings* NVStrings::upper()
{
  unsigned int count = size();
  if (count == 0) return new NVStrings(0);
  auto execpol                  = rmm::exec_policy(0);
  custring_view_array d_strings = pImpl->getStringsPtr();
  unsigned char* d_flags        = get_unicode_flags();
  unsigned short* d_cases       = get_charcases();
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (dstr) {
                         unsigned int bytes = 0;
                         for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
                           Char chr         = *itr;
                           unsigned int chw = custring_view::bytes_in_char(chr);
                           unsigned int uni = u82u(chr);
                           unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                           if (IS_LOWER(flg))
                             chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                           bytes += chw;
                         }
                         d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes, dstr->chars_count()));
                       }
                     });

  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  size_t* d_offsets = offsets.data().get();
  // do the thing
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (!dstr) return;
      char* buffer       = d_buffer + d_offsets[idx];
      char* ptr          = buffer;
      unsigned int bytes = 0;
      for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
        Char chr         = *itr;
        unsigned int uni = u82u(*itr);
        unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
        if (IS_LOWER(flg)) chr = u2u8(d_cases[uni]);
        int chw = custring_view::Char_to_char(chr, ptr);
        ptr += chw;
        bytes += chw;
      }
      d_results[idx] = custring_view::create_from(buffer, buffer, bytes);
    });
  //
  return rtn;
}

//
NVStrings* NVStrings::swapcase()
{
  unsigned int count = size();
  if (count == 0) return new NVStrings(0);
  auto execpol                  = rmm::exec_policy(0);
  custring_view_array d_strings = pImpl->getStringsPtr();
  unsigned char* d_flags        = get_unicode_flags();
  unsigned short* d_cases       = get_charcases();
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (dstr) {
                         unsigned int bytes = 0;
                         for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
                           Char chr         = *itr;
                           unsigned int chw = custring_view::bytes_in_char(chr);
                           unsigned int uni = u82u(chr);
                           unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                           if (IS_LOWER(flg) || IS_UPPER(flg))
                             chw = custring_view::bytes_in_char(u2u8(d_cases[uni]));
                           bytes += chw;
                         }
                         d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes, dstr->chars_count()));
                       }
                     });
  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the thing
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (dstr) {
        char* buffer       = d_buffer + d_offsets[idx];
        char* ptr          = buffer;
        unsigned int bytes = 0;
        for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
          Char chr         = *itr;
          unsigned int uni = u82u(*itr);
          unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
          if (IS_LOWER(flg) || IS_UPPER(flg)) chr = u2u8(d_cases[uni]);
          int chw = custring_view::Char_to_char(chr, ptr);
          ptr += chw;
          bytes += chw;
        }
        d_results[idx] = custring_view::create_from(buffer, buffer, bytes);
      }
    });
  //
  return rtn;
}

//
NVStrings* NVStrings::capitalize()
{
  unsigned int count = size();
  if (count == 0) return new NVStrings(0);
  auto execpol                  = rmm::exec_policy(0);
  custring_view_array d_strings = pImpl->getStringsPtr();
  unsigned char* d_flags        = get_unicode_flags();
  unsigned short* d_cases       = get_charcases();
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (dstr) {
                         unsigned int bytes = 0;
                         for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
                           Char chr         = *itr;
                           unsigned int chw = custring_view::bytes_in_char(chr);
                           unsigned int uni = u82u(chr);
                           unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                           if ((bytes && IS_UPPER(flg)) || (!bytes && IS_LOWER(flg))) {
                             uni = (uni <= 0x00FFF ? d_cases[uni] : uni);
                             chr = u2u8(uni);
                             chw = custring_view::bytes_in_char(chr);
                           }
                           bytes += chw;
                         }
                         d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes, dstr->chars_count()));
                       }
                     });
  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the thing
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (dstr) {
        char* buffer       = d_buffer + d_offsets[idx];
        char* ptr          = buffer;
        unsigned int bytes = 0;
        for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
          Char chr         = *itr;
          unsigned int uni = u82u(chr);
          unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
          if ((bytes && IS_UPPER(flg)) || (!bytes && IS_LOWER(flg))) {
            uni = (uni <= 0x00FFF ? d_cases[uni] : uni);
            chr = u2u8(uni);
          }
          unsigned int chw = custring_view::Char_to_char(chr, ptr);
          ptr += chw;
          bytes += chw;
        }
        d_results[idx] = custring_view::create_from(buffer, buffer, bytes);
      }
    });
  //
  return rtn;
}

// returns titlecase for each string
NVStrings* NVStrings::title()
{
  unsigned int count = size();
  if (count == 0) return new NVStrings(0);
  auto execpol                  = rmm::exec_policy(0);
  custring_view_array d_strings = pImpl->getStringsPtr();
  unsigned char* d_flags        = get_unicode_flags();
  unsigned short* d_cases       = get_charcases();
  // compute size of output buffer
  rmm::device_vector<size_t> lengths(count, 0);
  size_t* d_lengths = lengths.data().get();
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     [d_strings, d_flags, d_cases, d_lengths] __device__(unsigned int idx) {
                       custring_view* dstr = d_strings[idx];
                       if (dstr) {
                         int bytes     = 0;
                         bool bcapnext = true;
                         for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
                           Char chr         = *itr;
                           unsigned int uni = u82u(chr);
                           unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
                           if (!IS_ALPHA(flg)) {
                             bcapnext = true;
                             bytes += custring_view::bytes_in_char(chr);
                             continue;
                           }
                           if ((bcapnext && IS_LOWER(flg)) || (!bcapnext && IS_UPPER(flg)))
                             uni = (unsigned int)(uni <= 0x00FFFF ? d_cases[uni] : uni);
                           bcapnext = false;
                           bytes += custring_view::bytes_in_char(u2u8(uni));
                         }
                         d_lengths[idx] = ALIGN_SIZE(dstr->alloc_size(bytes, dstr->chars_count()));
                       }
                     });
  // create output object
  NVStrings* rtn = new NVStrings(count);
  char* d_buffer = rtn->pImpl->createMemoryFor(d_lengths);
  if (d_buffer == 0) return rtn;
  // create offsets
  rmm::device_vector<size_t> offsets(count, 0);
  thrust::exclusive_scan(execpol->on(0), lengths.begin(), lengths.end(), offsets.begin());
  // do the title thing
  custring_view_array d_results = rtn->pImpl->getStringsPtr();
  size_t* d_offsets             = offsets.data().get();
  thrust::for_each_n(
    execpol->on(0),
    thrust::make_counting_iterator<unsigned int>(0),
    count,
    [d_strings, d_flags, d_cases, d_buffer, d_offsets, d_results] __device__(unsigned int idx) {
      custring_view* dstr = d_strings[idx];
      if (dstr) {
        char* buffer  = d_buffer + d_offsets[idx];
        char* ptr     = buffer;
        int bytes     = 0;
        bool bcapnext = true;
        for (auto itr = dstr->begin(); (itr != dstr->end()); itr++) {
          Char chr         = *itr;
          unsigned int uni = u82u(chr);
          unsigned int flg = (uni <= 0x00FFFF ? d_flags[uni] : 0);
          if (!IS_ALPHA(flg))
            bcapnext = true;
          else {
            if ((bcapnext && IS_LOWER(flg)) || (!bcapnext && IS_UPPER(flg))) {
              uni = (unsigned int)(uni <= 0x00FFFF ? d_cases[uni] : uni);
              chr = u2u8(uni);
            }
            bcapnext = false;
          }
          int chw = custring_view::Char_to_char(chr, ptr);
          bytes += chw;
          ptr += chw;
        }
        d_results[idx] = custring_view::create_from(buffer, buffer, bytes);
      }
    });
  //
  return rtn;
}
