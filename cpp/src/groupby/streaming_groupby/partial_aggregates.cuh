/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "merge_aggregates.cuh"

#include <cudf/types.hpp>

#include <cstdint>

namespace cudf::groupby {

inline constexpr int partial_aggregation_block_size = 256;

/**
 * @brief Merges shard-major partial states into a destination group's first shard.
 *
 * A null group map denotes the identity mapping. The scalar merge operation is shared with
 * streaming merge, so counts are summed and already-squared SUM_OF_SQUARES values are not squared.
 */
struct merge_partial_aggs_fn {
  size_type const* group_targets;
  size_type source_capacity;
  size_type source_groups;
  size_type source_replicas;
  aggregation::Kind const* aggs;
  table_device_view source_values;
  mutable_table_device_view target_values;

  __device__ void operator()(int64_t idx) const
  {
    if (source_groups == 0) { return; }
    auto const num_rows = static_cast<int64_t>(source_groups) * source_replicas;
    auto const position = idx % num_rows;
    auto const group    = static_cast<size_type>(position % source_groups);
    auto const shard    = static_cast<size_type>(position / source_groups);
    // Capacity is the physical pitch; iterate only live groups in every shard.
    auto const source_row = shard * source_capacity + group;
    auto const target     = group_targets == nullptr ? group : group_targets[group];
    if (target == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { return; }
    auto const column  = static_cast<size_type>(idx / num_rows);
    auto const& values = source_values.column(column);
    cudf::detail::dispatch_type_and_aggregation(values.type(),
                                                aggs[column],
                                                merge_element_aggregator{},
                                                target_values.column(column),
                                                target,
                                                values,
                                                source_row);
  }
};

}  // namespace cudf::groupby
