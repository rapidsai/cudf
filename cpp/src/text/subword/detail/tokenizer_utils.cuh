/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "text/subword/detail/cp_data.h"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace nvtext {
namespace detail {

constexpr int THREADS_PER_BLOCK = 64;

/**
 * @brief In-place update of offsets values.
 *
 * In the `d_chars_up_to_idx`, the last character of each string is basically
 * the offset (i.e. the number of characters) in that string.
 *
 * Example
 * @code{.pseudo}
 * // 3 strings with sizes 5,4,2
 * d_offsets = [0,5,9,11]
 * // code points generated per character (as offsets)
 * // 2nd string has an extra code point at its first char
 * d_chars_up_to_idx = [1,2,3,4,5,6,8,9,10,11,12]
 * d_chars_up_to_idx[d_offsets[1-3]] is [5,10,12]
 * => d_offsets becomes [0,5,10,12]
 * @endcode
 */
struct update_strings_lengths_fn {
  uint32_t const* d_chars_up_to_idx;
  int64_t* d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    auto const offset = d_offsets[idx];
    d_offsets[idx]    = offset > 0 ? d_chars_up_to_idx[offset - 1] : 0;
  }
};

/**
 * @brief Retrieve the code point metadata table.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_uvector<codepoint_metadata_type> get_codepoint_metadata(rmm::cuda_stream_view stream);

/**
 * @brief Retrieve the auxiliary code point metadata table.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_uvector<aux_codepoint_data_type> get_aux_codepoint_data(rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace nvtext
