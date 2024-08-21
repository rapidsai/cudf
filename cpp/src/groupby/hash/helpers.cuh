/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <cuco/static_set.cuh>

namespace cudf::groupby::detail::hash {

CUDF_HOST_DEVICE constexpr std::size_t round_to_multiple_of_8(std::size_t num)
{
  std::size_t constexpr base = 8;
  return cudf::util::div_rounding_up_safe(num, base) * base;
}

// TODO: similar to `contains_table`, using larger CG size like 2 or 4 for nested
// types and `cg_size = 1`for flat data to improve performance
/// Number of threads to handle each input element
CUDF_HOST_DEVICE auto constexpr GROUPBY_CG_SIZE = 1;
/// Number of slots per thread
CUDF_HOST_DEVICE auto constexpr GROUPBY_WINDOW_SIZE = 1;

using probing_scheme_type = cuco::linear_probing<
  GROUPBY_CG_SIZE,
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>>;

}  // namespace cudf::groupby::detail::hash
