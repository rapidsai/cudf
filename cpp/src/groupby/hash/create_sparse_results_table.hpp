/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes and returns a device vector containing all populated keys in
 * `key_set`.
 *
 * @tparam SetType Type of the key hash set
 *
 * @param key_set Key hash set
 * @param populated_keys Array of unique keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return An array of unique keys contained in `key_set`
 */
template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            rmm::cuda_stream_view stream);

// make table that will hold sparse results
template <typename GlobalSetType>
cudf::table create_sparse_results_table(cudf::table_view const& flattened_values,
                                        cudf::aggregation::Kind const* d_agg_kinds,
                                        host_span<cudf::aggregation::Kind const> agg_kinds,
                                        bool direct_aggregations,
                                        GlobalSetType const& global_set,
                                        rmm::device_uvector<cudf::size_type>& populated_keys,
                                        rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
