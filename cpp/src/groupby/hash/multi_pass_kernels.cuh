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

namespace cudf {
namespace detail {

// TODO it could be merged to aggregation.cuh
template <typename Source,
          aggregation::Kind k,
          bool target_has_nulls,
          bool source_has_nulls,
          typename Enable = void>
struct update_target_element2 {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index,
                             table_device_view dependent_values,
                             size_t const* dependent_offset,
                             size_t i,
                             size_type ddof) const noexcept
  {
    release_assert(false and "Invalid source type and aggregation combination.");
  }
};

template <typename Source, bool target_has_nulls, bool source_has_nulls>
struct update_target_element2<Source,
                              aggregation::MEAN,
                              target_has_nulls,
                              source_has_nulls,
                              std::enable_if_t<is_numeric<Source>() && !is_fixed_point<Source>()>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index,
                             table_device_view dependent_values,
                             size_t const* dependent_offset,
                             size_t i,
                             size_type ddof) const noexcept
  {
    using Target    = target_type_t<Source, aggregation::MEAN>;
    using SumType   = target_type_t<Source, aggregation::SUM>;
    using CountType = target_type_t<Source, aggregation::COUNT_VALID>;
    if (source_has_nulls and source.is_null(source_index)) { return; }
    auto const sum       = dependent_values.column(dependent_offset[i]);
    auto const count     = dependent_values.column(dependent_offset[i] + 1);
    CountType group_size = count.element<CountType>(target_index);
    // prevent divide by zero error
    if (group_size == 0) return;

    // Target mean = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    Target mean = static_cast<Target>(sum.element<Source>(source_index)) / group_size;
    atomicAdd(&target.element<Target>(target_index), mean);

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

// template class specialization for STD, VARIANCE
template <typename Source, aggregation::Kind k, bool target_has_nulls, bool source_has_nulls>
struct update_target_element2<
  Source,
  k,
  target_has_nulls,
  source_has_nulls,
  std::enable_if_t<is_numeric<Source>() && !is_fixed_point<Source>() &&
                   (k == aggregation::VARIANCE or k == aggregation::STD)>> {
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index,
                             table_device_view dependent_values,
                             size_t const* dependent_offset,
                             size_t i,
                             size_type ddof) const noexcept
  {
    using Target    = target_type_t<Source, aggregation::VARIANCE>;
    using SumType   = target_type_t<Source, aggregation::SUM>;
    using CountType = target_type_t<Source, aggregation::COUNT_VALID>;

    if (source_has_nulls and source.is_null(source_index)) { return; }
    auto const sum       = dependent_values.column(dependent_offset[i]);
    auto const count     = dependent_values.column(dependent_offset[i] + 1);
    CountType group_size = count.element<CountType>(target_index);
    if (group_size == 0 or group_size - ddof <= 0) return;

    auto x        = static_cast<Target>(source.element<Source>(source_index));
    auto mean     = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    Target result = (x - mean) * (x - mean) / (group_size - ddof);
    atomicAdd(&target.element<Target>(target_index), result);
    // STD sqrt is applied in finalize()

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
};

template <bool target_has_nulls = true, bool source_has_nulls = true>
struct elementwise_aggregator_pass2 {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index,
                             table_device_view dependent_values,
                             size_t const* dependent_offset,
                             size_t i,
                             size_type ddof) const noexcept
  {
    update_target_element2<Source, k, target_has_nulls, source_has_nulls>{}(
      target, target_index, source, source_index, dependent_values, dependent_offset, i, ddof);
  }
};

template <bool target_has_nulls = true, bool source_has_nulls = true>
__device__ inline void aggregate_row_pass2(mutable_table_device_view target,
                                           size_type target_index,
                                           table_device_view source,
                                           size_type source_index,
                                           aggregation::Kind const* aggs,
                                           table_device_view dependent_values,
                                           size_t const* dependent_offset,
                                           size_type const* ddofs)
{
  for (auto i = 0; i < target.num_columns(); ++i) {
    dispatch_type_and_aggregation(
      source.column(i).type(),
      aggs[i],
      elementwise_aggregator_pass2<target_has_nulls, source_has_nulls>{},
      target.column(i),
      target_index,
      source.column(i),
      source_index,
      dependent_values,
      dependent_offset,
      i,
      ddofs[i]);
  }
}

}  // namespace detail
}  // namespace cudf
