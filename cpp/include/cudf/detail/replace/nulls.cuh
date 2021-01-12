/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

/**
 * @brief Functor used by `inclusive_scan` to determine the index to gather from in
 *        the result column. When current row in input column is NULL, return previous
 *        accumulated index, otherwise return the current index. The second element in
 *        the return tuple is discarded.
 */
struct replace_policy_functor {
  __device__ thrust::tuple<cudf::size_type, bool> operator()(
    thrust::tuple<cudf::size_type, bool> const& lhs,
    thrust::tuple<cudf::size_type, bool> const& rhs);
};

}  // namespace detail
}  // namespace cudf
