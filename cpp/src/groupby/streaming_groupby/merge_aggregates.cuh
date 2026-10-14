/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/table/table_device_view.cuh>

namespace cudf::groupby {

/**
 * @brief Element aggregator for merging intermediate results.
 */
struct merge_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (!cudf::detail::is_valid_aggregation<Source, k>()) {
      return;
    } else {
      if constexpr (k != aggregation::COUNT_ALL) {
        if (source.is_null(source_index)) { return; }
      }
      if constexpr (!(k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL)) {
        if (target.is_null(target_index)) { target.set_valid(target_index); }
      }

      if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 source.element<Target>(source_index));
      } else if constexpr (k == aggregation::SUM_OF_SQUARES) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 static_cast<Target>(source.element<Source>(source_index)));
      } else {
        cudf::detail::update_target_element<Source, k>{}(
          target, target_index, source, source_index);
      }
    }
  }
};

struct merge_single_pass_aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view source_values;
  mutable_table_device_view target_values;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = source_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    if (auto const target_row_idx = target_indices[source_row_idx];
        target_row_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      auto const col_idx     = static_cast<size_type>(idx / num_rows);
      auto const& source_col = source_values.column(col_idx);
      auto const& target_col = target_values.column(col_idx);
      cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                  aggs[col_idx],
                                                  merge_element_aggregator{},
                                                  target_col,
                                                  target_row_idx,
                                                  source_col,
                                                  source_row_idx);
    }
  }
};

}  // namespace cudf::groupby
