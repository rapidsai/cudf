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

// NOTE: These are cudf headers. Please be cautious.
// Using anything from these headers besides macros or typedefs
// will not work because this module is built before libcudf
// and therefore will not be able to link to any functions there.
// This module will be reworked appropriately in the future.
#include <cudf/utilities/error.hpp>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "../custring_view.cuh"
#include "../util.h"

struct stemmer_base_fn {
  custring_view* d_vowels;
  Char y_char;

  stemmer_base_fn(custring_view* d_vowels, Char y_char) : d_vowels(d_vowels), y_char(y_char) {}

  __device__ bool is_consonant(custring_view* dstr, int index) const
  {
    Char ch = dstr->at(index);
    if (d_vowels->find(ch) >= 0) return false;
    if ((ch != y_char) || (index == 0)) return true;
    ch = dstr->at(index - 1);        // only if previous char
    return d_vowels->find(ch) >= 0;  // is not a consonant
  }
};

struct porter_stemmer_measure_fn : public stemmer_base_fn {
  custring_view_array d_strings;
  unsigned int* d_results;

  porter_stemmer_measure_fn(custring_view* d_vowels,
                            Char y_char,
                            custring_view_array d_strings,
                            unsigned int* d_results)
    : stemmer_base_fn(d_vowels, y_char), d_strings(d_strings), d_results(d_results)
  {
  }

  __device__ void operator()(unsigned int idx)
  {
    custring_view* dstr = d_strings[idx];
    if (!dstr) return;
    unsigned int vcs = 0;
    bool vowel_run   = !is_consonant(dstr, 0);
    for (auto itr = dstr->begin(); itr != dstr->end(); itr++) {
      if (is_consonant(dstr, itr.position())) {
        if (vowel_run) vcs++;
        vowel_run = false;
      } else
        vowel_run = true;
    }
    d_results[idx] = vcs;
  }
};

unsigned int NVText::porter_stemmer_measure(
  NVStrings& strs, const char* vowels, const char* y_char, unsigned int* results, bool bdevmem)
{
  unsigned int count = strs.size();
  if (count == 0) return 0;  // nothing to do
  auto execpol = rmm::exec_policy(0);
  // setup results vector
  unsigned int* d_results = results;
  if (!bdevmem) d_results = device_alloc<unsigned int>(count, 0);
  if (vowels == nullptr) vowels = "aeiou";
  custring_view* d_vowels = custring_from_host(vowels);
  if (y_char == nullptr) y_char = "y";
  Char char_y;
  custring_view::char_to_Char(y_char, char_y);

  // get the string pointers
  rmm::device_vector<custring_view*> strings(count, nullptr);
  custring_view** d_strings = strings.data().get();
  strs.create_custring_index(d_strings);

  // do the measure
  thrust::for_each_n(execpol->on(0),
                     thrust::make_counting_iterator<unsigned int>(0),
                     count,
                     porter_stemmer_measure_fn{d_vowels, char_y, d_strings, d_results});

  // done
  if (!bdevmem) {
    CUDA_TRY(
      cudaMemcpyAsync(results, d_results, count * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    RMM_FREE(d_results, 0);
  }
  RMM_FREE(d_vowels, 0);
  return 0;
}

//
unsigned int is_letter(NVStrings& strs,
                       const char* vowels,
                       const char* y_char,
                       NVText::letter_type ltype,
                       int index,
                       int* d_indices,
                       bool* results,
                       bool bdevmem)
{
  unsigned int count = strs.size();
  if (count == 0) return 0;  // nothing to do
  auto execpol = rmm::exec_policy(0);
  // setup results vector
  bool* d_results = results;
  if (!bdevmem) d_results = device_alloc<bool>(count, 0);
  if (vowels == nullptr) vowels = "aeiou";
  custring_view* d_vowels = custring_from_host(vowels);
  if (y_char == nullptr) y_char = "y";
  Char char_y;
  custring_view::char_to_Char(y_char, char_y);

  // get the string pointers
  rmm::device_vector<custring_view*> strings(count, nullptr);
  custring_view** d_strings = strings.data().get();
  strs.create_custring_index(d_strings);

  //
  stemmer_base_fn pfn{d_vowels, char_y};
  thrust::transform(execpol->on(0),
                    thrust::make_counting_iterator<unsigned int>(0),
                    thrust::make_counting_iterator<unsigned int>(count),
                    d_results,
                    [d_strings, pfn, ltype, index, d_indices] __device__(unsigned int idx) {
                      custring_view* d_str = d_strings[idx];
                      if (!d_str) return false;
                      int position = index;
                      if (d_indices) position = d_indices[idx];
                      int length = static_cast<int>(d_str->length());
                      if ((position >= length) || (position < -length)) return false;
                      position =
                        (position + length) % length;  // handles positive or negative index
                      return pfn.is_consonant(d_str, position) ? ltype == NVText::consonant
                                                               : ltype == NVText::vowel;
                    });

  // done
  if (!bdevmem) {
    CUDA_TRY(cudaMemcpyAsync(results, d_results, count * sizeof(bool), cudaMemcpyDeviceToHost))
    RMM_FREE(d_results, 0);
  }
  RMM_FREE(d_vowels, 0);
  return 0;
}

// check individual characters are vowels or consonants
unsigned int NVText::is_letter(NVStrings& strs,
                               const char* vowels,
                               const char* y_char,
                               NVText::letter_type ltype,
                               int position,
                               bool* results,
                               bool bdevmem)
{
  return ::is_letter(strs, vowels, y_char, ltype, position, nullptr, results, bdevmem);
}

//
unsigned int NVText::is_letter(NVStrings& strs,
                               const char* vowels,
                               const char* y_char,
                               NVText::letter_type ltype,
                               int* d_indices,
                               bool* results,
                               bool bdevmem)
{
  return ::is_letter(strs, vowels, y_char, ltype, 0, d_indices, results, bdevmem);
}
