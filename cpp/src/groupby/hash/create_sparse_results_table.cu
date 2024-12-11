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

#include "create_sparse_results_table.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            rmm::cuda_stream_view stream)
{
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
}

// make table that will hold sparse results
template <typename GlobalSetType>
cudf::table create_sparse_results_table(cudf::table_view const& flattened_values,
                                        cudf::aggregation::Kind const* d_agg_kinds,
                                        std::vector<cudf::aggregation::Kind> agg_kinds,
                                        bool direct_aggregations,
                                        GlobalSetType const& global_set,
                                        rmm::device_uvector<cudf::size_type>& populated_keys,
                                        rmm::cuda_stream_view stream)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<cudf::column>> sparse_columns;
  std::transform(flattened_values.begin(),
                 flattened_values.end(),
                 agg_kinds.begin(),
                 std::back_inserter(sparse_columns),
                 [stream](auto const& col, auto const& agg) {
                   auto const nullable =
                     (agg == cudf::aggregation::COUNT_VALID or agg == cudf::aggregation::COUNT_ALL)
                       ? false
                       : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or
                          agg == cudf::aggregation::STD);
                   auto const mask_flag =
                     (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;
                   auto const col_type = cudf::is_dictionary(col.type())
                                           ? cudf::dictionary_column_view(col).keys().type()
                                           : col.type();
                   return make_fixed_width_column(
                     cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
                 });
  cudf::table sparse_table(std::move(sparse_columns));
  // If no direct aggregations, initialize the sparse table
  // only for the keys inserted in global hash set
  if (!direct_aggregations) {
    auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);
    extract_populated_keys(global_set, populated_keys, stream);
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      populated_keys.size(),
      initialize_sparse_table{populated_keys.data(), *d_sparse_table, d_agg_kinds});
  }
  // Else initialize the whole table
  else {
    cudf::mutable_table_view sparse_table_view = sparse_table.mutable_view();
    cudf::detail::initialize_with_identity(sparse_table_view, agg_kinds, stream);
  }
  return sparse_table;
}

template void extract_populated_keys<global_set_t>(
  global_set_t const& key_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template void extract_populated_keys<nullable_global_set_t>(
  nullable_global_set_t const& key_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template cudf::table create_sparse_results_table<global_set_t>(
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  std::vector<cudf::aggregation::Kind> agg_kinds,
  bool direct_aggregations,
  global_set_t const& global_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template cudf::table create_sparse_results_table<nullable_global_set_t>(
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  std::vector<cudf::aggregation::Kind> agg_kinds,
  bool direct_aggregations,
  nullable_global_set_t const& global_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
