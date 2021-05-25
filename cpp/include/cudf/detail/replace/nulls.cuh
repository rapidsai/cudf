/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <thrust/functional.h>

namespace cudf {
namespace detail {

using idx_valid_pair_t = thrust::tuple<cudf::size_type, bool>;

/**
 * @brief Functor used by `replace_nulls(replace_policy)` to determine the index to gather from in
 * the result column.
 *
 * Binary functor passed to `inclusive_scan` or `inclusive_scan_by_key`. Arguments are a tuple of
 * index and validity of a row. Returns a tuple of current index and a discarded boolean if current
 * row is valid, otherwise a tuple of the nearest non-null row index and a discarded boolean.
 */
struct replace_policy_functor {
  __device__ idx_valid_pair_t operator()(idx_valid_pair_t const& lhs, idx_valid_pair_t const& rhs)
  {
    return thrust::get<1>(rhs) ? thrust::make_tuple(thrust::get<0>(rhs), true)
                               : thrust::make_tuple(thrust::get<0>(lhs), true);
  }
};

}  // namespace detail
}  // namespace cudf
