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

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

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
                                        std::vector<cudf::aggregation::Kind> agg_kinds,
                                        bool direct_aggregations,
                                        GlobalSetType const& global_set,
                                        rmm::device_uvector<cudf::size_type>& populated_keys,
                                        rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
