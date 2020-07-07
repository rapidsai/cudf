/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <vector>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

using codepoint_metadata_type = uint32_t;
using aux_codepoint_data_type = uint64_t;

namespace nvtext {
namespace detail {

struct ptr_length_pair {
  uint32_t* gpu_ptr{};
  size_t length{};
};

/**
 * @brief Performs text cleaning for the tokenizers.
 *
 * Every instantiation of this class will transfer the meta data over to the GPU.
 * It is advised to create one class and reuse that class as needed.
 *
 * Converts characters to lowercase, adds spaces around punctuation and multi-byte
 * characters, strips accents from letters in the text and standardizes whitespace
 * characters to all be the code point for the " " literal.
 */
class data_normalizer {
 public:
  /**
   * @brief Transfer to the GPU the metadata needed to normalize characters.
   *
   * @param max_num_strings Maximum number of input strings for instantiating the normalizer.
   *        Used to allocate temporary working memory on device.
   *        If the input contains a larger number of strings, behavior is undefined.
   * @param max_num_chars Maximum number of characters for instantiating the normalizer.
   *        Used to allocate temporary working memory on device.
   *        If the input contains a larger number of characters, behavior is undefined.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param do_lower_case If true, the normalizer will convert uppercase characters in the
   *        input stream to lower case and strip accents from those characters.
   *        If false, accented and uppercase characters are not transformed.
   */
  data_normalizer(uint32_t max_num_strings,
                  uint32_t max_num_chars,
                  cudaStream_t stream,
                  bool do_lower_case = true);

  /**
   * @brief Normalize a vector of strings.
   *
   * If `do_lower_case` is true, this function will convert each character to lowercase
   * and strip accents from the characters. If false it will do all other conversions
   * in the class description except lower-casing and punctuation stripping.
   *
   * The result of this function returns two pointers to GPU data along with their lengths.
   * The first pointer is to a contiguous array of unicode code points corresponding to the
   * characters in the text after running normalization. The second pointer is to the
   * offsets of the strings in the code point array. That is, string `i` starts at
   * `result.second.gpu_ptr[i]`.
   * This array will always be of length `num_strings + 1` since we need one entry
   * for each input and a last entry which has the total number of bytes.
   *
   * @param d_strings A vector of strings which MUST be encoded in the UTF-8 format.
   * @param d_offsets A vector of byte offsets to the beginning of individual strings in
   *        the `d_strings` parameter.
   * @param num_strings The number of strings identified in `d_strings`.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @return Two pointers to GPU data along with their lengths. The first is a pointer
   *         to the code points array and the second is a pointer to the offsets
   *         used to locate the code points for each string.
   */
  std::pair<ptr_length_pair, ptr_length_pair> normalize(char const* d_strings,
                                                        uint32_t const* d_offsets,
                                                        uint32_t num_strings,
                                                        cudaStream_t stream);

 private:
  bool const do_lower_case;
  codepoint_metadata_type const* d_cp_metadata;
  aux_codepoint_data_type const* d_aux_table;

  // working memory for the normalization logic
  rmm::device_uvector<unsigned char> device_strings;
  rmm::device_uvector<uint32_t> device_strings_offsets;
  rmm::device_uvector<uint32_t> device_code_points;
  rmm::device_uvector<uint32_t> device_chars_per_thread;
  rmm::device_uvector<size_t> cub_temp_storage;
  rmm::device_uvector<uint32_t> device_num_selected;
  size_t max_cub_storage_bytes;
};

}  // namespace detail
}  // namespace nvtext
