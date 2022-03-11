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
 * @brief bool generator with given probability [0.0 - 1.0] of returning true.
 *
 */
struct bool_generator {
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> dist;
  float probability_true;
  bool_generator(unsigned seed, float probability_true)
    : engine(seed), dist{0, 1}, probability_true{probability_true}
  {
  }

  __device__ bool operator()(size_t n)
  {
    engine.discard(n);
    return dist(engine) < probability_true;
  }
};

std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(cudf::size_type size,
                                                                       float null_probability,
                                                                       unsigned seed)
{
  if (null_probability < 0.0f) {
    return {rmm::device_buffer{}, 0};
  } else if (null_probability == 0.0f) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_NULL), size};
  } else if (null_probability >= 1.0f) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_VALID), 0};
  } else {
    return cudf::detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                  thrust::make_counting_iterator<cudf::size_type>(size),
                                  bool_generator{seed, 1.0f - null_probability});
  }
};
