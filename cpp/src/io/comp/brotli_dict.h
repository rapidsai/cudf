/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

/*
 * Portions of this file are derived from Google's Brotli project at
 * https://github.com/google/brotli, original license text below.
 */

/* Copyright 2013 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/*
Copyright(c) 2009, 2010, 2013 - 2016 by the Brotli Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#pragma once

#include <cstdint>

namespace cudf {
namespace io {
struct brotli_dictionary_s {
  /**
   * Number of bits to encode index of dictionary word in a bucket.
   *
   * Specification: Appendix A. Static Dictionary Data
   *
   * Words in a dictionary are bucketed by length.
   * @c 0 means that there are no words of a given length.
   * Dictionary consists of words with length of [4..24] bytes.
   * Values at [0..3] and [25..31] indices should not be addressed.
   */
  uint8_t size_bits_by_length[32];

  /* assert(offset[i + 1] == offset[i] + (bits[i] ? (i << bits[i]) : 0)) */
  uint32_t offsets_by_length[32];

  /* Data array should obey to size_bits_by_length values.
    Specified size matches default (RFC 7932) dictionary.
    Its size is also equal to offsets_by_length[31] */
  uint8_t data[122784];
};

#define BROTLI_MIN_DICTIONARY_WORD_LENGTH 4
#define BROTLI_MAX_DICTIONARY_WORD_LENGTH 24

const brotli_dictionary_s *get_brotli_dictionary(void);

}  // namespace io
}  // namespace cudf
