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

#include <stdint.h>
#include <vector>

namespace nvtext {
namespace detail {

constexpr int THREADS_PER_BLOCK = 64;

/**
 * @brief A selection op for cub to get elements from an array not equal to a certain value.
 *
 * See https://nvlabs.github.io/cub/structcub_1_1_device_partition.html for an example of
 * this struct.
 */
struct NotEqual {
  uint32_t const val_to_omit;

  __host__ __device__ NotEqual(uint32_t const val_to_omit) : val_to_omit(val_to_omit) {}

  __host__ __device__ bool operator()(uint32_t const& a) const { return (a != val_to_omit); }
};

/**
 * @brief In-place update of lengths values.
 */
struct update_strings_lengths_fn {
  uint32_t const* d_chars_up_to_idx;
  uint32_t* d_lengths;
  __device__ void operator()(uint32_t idx)
  {
    d_lengths[idx] = d_chars_up_to_idx[d_lengths[idx] - 1];
  }
};

}  // namespace detail
}  // namespace nvtext
