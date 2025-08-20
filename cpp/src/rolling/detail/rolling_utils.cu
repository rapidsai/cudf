/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "rolling.hpp"
#include "rolling_operators.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace detail {
namespace {
struct is_supported_rolling_aggregation_impl {
  template <typename T, aggregation::Kind kind>
  constexpr bool operator()() const noexcept
  {
    return (kind == aggregation::Kind::LEAD || kind == aggregation::Kind::LAG ||
            kind == aggregation::Kind::COLLECT_LIST || kind == aggregation::Kind::COLLECT_SET) ||
           corresponding_rolling_operator<T, kind>::type::is_supported();
  }
};
}  // namespace

bool is_valid_rolling_aggregation(data_type source, aggregation::Kind kind)
{
  return dispatch_type_and_aggregation(source, kind, is_supported_rolling_aggregation_impl{});
}
}  // namespace detail
}  // namespace cudf
