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

// A selection op for cub to get elements from an array not equal to a certain value. See
// https://nvlabs.github.io/cub/structcub_1_1_device_partition.html for an example of this
// struct.
struct NotEqual {
  uint32_t val_to_omit;

  __host__ __device__ __forceinline__ NotEqual(uint32_t val_to_omit) : val_to_omit(val_to_omit) {}

  __host__ __device__ __forceinline__ bool operator()(const uint32_t& a) const
  {
    return (a != val_to_omit);
  }
};

/*

*/
static __global__ void update_sentence_lengths(uint32_t* old_lengths,
                                               uint32_t* chars_up_to_idx,
                                               size_t num_sentences)
{
  uint32_t sen_for_thread = threadIdx.x + blockDim.x * blockIdx.x + 1;

  if (sen_for_thread <= num_sentences) {
    old_lengths[sen_for_thread] = chars_up_to_idx[old_lengths[sen_for_thread] - 1];
  }
}

}  // namespace detail
}  // namespace nvtext
