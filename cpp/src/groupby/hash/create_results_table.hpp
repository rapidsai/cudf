/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::groupby::detail::hash {

// TODO
rmm::device_uvector<size_type> find_output_indices(device_span<size_type> key_indices,
                                                   device_span<size_type const> unique_indices,
                                                   rmm::cuda_stream_view stream);

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
                            rmm::device_uvector<size_type>& populated_keys,
                            rmm::cuda_stream_view stream);

table create_results_table(cudf::size_type output_size,
                           table_view const& flattened_values,
                           host_span<aggregation::Kind const> agg_kinds,
                           rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
