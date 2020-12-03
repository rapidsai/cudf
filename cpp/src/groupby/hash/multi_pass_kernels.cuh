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

#include <cmath>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace detail {

template <typename Map, bool target_has_nulls = true, bool source_has_nulls = true>
struct var_hash_functor {
  Map const map;
  bitmask_type const* __restrict__ row_bitmask;
  mutable_column_device_view target;
  column_device_view source;
  column_device_view sum;
  column_device_view count;
  size_type ddof;
  var_hash_functor(Map const map,
                   bitmask_type const* row_bitmask,
                   mutable_column_device_view target,
                   column_device_view source,
                   column_device_view sum,
                   column_device_view count,
                   size_type ddof)
    : map(map),
      row_bitmask(row_bitmask),
      target(target),
      source(source),
      sum(sum),
      count(count),
      ddof(ddof)
  {
  }

  template <typename Source>
  constexpr static bool is_supported()
  {
    return is_numeric<Source>() && !is_fixed_point<Source>();
  }

  template <typename Source>
  __device__ std::enable_if_t<!is_supported<Source>()> operator()(size_type source_index,
                                                                  size_type target_index) noexcept
  {
    release_assert(false and "Invalid source type for std, var aggregation combination.");
  }

  template <typename Source>
  __device__ std::enable_if_t<is_supported<Source>()> operator()(size_type source_index,
                                                                 size_type target_index) noexcept
  {
    using Target    = target_type_t<Source, aggregation::VARIANCE>;
    using SumType   = target_type_t<Source, aggregation::SUM>;
    using CountType = target_type_t<Source, aggregation::COUNT_VALID>;

    if (source_has_nulls and source.is_null(source_index)) return;
    CountType group_size = count.element<CountType>(target_index);
    if (group_size == 0 or group_size - ddof <= 0) return;

    auto x        = static_cast<Target>(source.element<Source>(source_index));
    auto mean     = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    Target result = (x - mean) * (x - mean) / (group_size - ddof);
    atomicAdd(&target.element<Target>(target_index), result);
    // STD sqrt is applied in finalize()

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
  __device__ inline void operator()(size_type source_index)
  {
    if (row_bitmask == nullptr or cudf::bit_is_set(row_bitmask, source_index)) {
      auto result       = map.find(source_index);
      auto target_index = result->second;
      type_dispatcher(source.type(), *this, source_index, target_index);
    }
  }
};

}  // namespace detail
}  // namespace cudf
