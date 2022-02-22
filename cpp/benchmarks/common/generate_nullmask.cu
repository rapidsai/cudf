/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "generate_input.hpp"

#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>

#include <thrust/random.h>

/**
 * @brief valid bit generator with given probability [0.0 - 1.0]
 *
 */
struct valid_generator {
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> dist;
  float valid_prob;
  valid_generator(thrust::minstd_rand engine, float valid_probability)
    : engine(engine), dist{0, 1}, valid_prob{valid_probability}
  {
  }
  valid_generator(unsigned seed, float valid_probability)
    : engine(seed), dist{0, 1}, valid_prob{valid_probability}
  {
  }

  __device__ bool operator()(size_t n)
  {
    engine.discard(n);
    return dist(engine) < valid_prob;
  }
};

std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(cudf::size_type size,
                                                                       float null_probability,
                                                                       unsigned seed)
{
  if (null_probability < 0.0f) {
    return {rmm::device_buffer{}, 0};
  } else if (null_probability >= 1.0f or null_probability == 0.0f) {
    return {
      cudf::create_null_mask(
        size, null_probability >= 1.0f ? cudf::mask_state::ALL_VALID : cudf::mask_state::ALL_NULL),
      null_probability >= 1.0f ? size : 0};
  } else {
    return cudf::detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                  thrust::make_counting_iterator<cudf::size_type>(size),
                                  valid_generator{seed, 1.0f - null_probability});
  }
};
