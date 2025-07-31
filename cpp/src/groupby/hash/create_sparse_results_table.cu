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

#include "create_sparse_results_table.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
namespace {
/**
 * @brief Functor to create sparse result columns for hash-based groupby aggregations
 *
 * This functor handles the creation of appropriately typed and sized columns for each
 * aggregation, including special handling for SUM_WITH_OVERFLOW which requires a struct column.
 */
struct sparse_column_creator {
  rmm::cuda_stream_view stream;

  explicit sparse_column_creator(rmm::cuda_stream_view stream) : stream(stream) {}

  std::unique_ptr<cudf::column> operator()(cudf::column_view const& col,
                                           cudf::aggregation::Kind const& agg) const
  {
    auto const nullable =
      (agg == cudf::aggregation::COUNT_VALID or agg == cudf::aggregation::COUNT_ALL)
        ? false
        : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or agg == cudf::aggregation::STD);
    auto const mask_flag = (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;
    auto const col_type  = cudf::is_dictionary(col.type())
                             ? cudf::dictionary_column_view(col).keys().type()
                             : col.type();

    // Special handling for SUM_WITH_OVERFLOW which needs a struct column
    if (agg == cudf::aggregation::SUM_WITH_OVERFLOW) {
      // Lambda to create empty columns for better readability
      auto make_empty_column = [&stream = this->stream](cudf::type_id type_id,
                                                        cudf::size_type size,
                                                        cudf::mask_state mask_state) {
        return make_fixed_width_column(cudf::data_type{type_id}, size, mask_state, stream);
      };

      // Lambda to create children for SUM_WITH_OVERFLOW struct column
      auto make_children = [&make_empty_column](cudf::size_type size, cudf::mask_state mask_state) {
        std::vector<std::unique_ptr<cudf::column>> children;
        // Create sum child column (int64_t) - no null mask needed, struct-level mask handles
        // nullability
        children.push_back(
          make_empty_column(cudf::type_id::INT64, size, cudf::mask_state::UNALLOCATED));
        // Create overflow child column (bool) - no null mask needed, only value matters
        children.push_back(
          make_empty_column(cudf::type_id::BOOL8, size, cudf::mask_state::UNALLOCATED));
        return children;
      };

      if (col.size() == 0) {
        // For empty columns, create empty struct column manually
        auto children = make_children(0, cudf::mask_state::UNALLOCATED);
        return create_structs_hierarchy(0, std::move(children), 0, {}, stream);
      } else {
        auto children = make_children(col.size(), mask_flag);

        // Create struct column with the children
        // For SUM_WITH_OVERFLOW, make struct nullable if input has nulls (same as other
        // aggregations)
        if (nullable) {
          // Start with ALL_NULL, results will be marked valid during aggregation
          auto null_mask  = cudf::create_null_mask(col.size(), cudf::mask_state::ALL_NULL, stream);
          auto null_count = col.size();  // All null initially
          return create_structs_hierarchy(
            col.size(), std::move(children), null_count, std::move(null_mask), stream);
        } else {
          return create_structs_hierarchy(col.size(), std::move(children), 0, {}, stream);
        }
      }
    } else {
      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
    }
  }
};
}  // anonymous namespace

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
                                        host_span<cudf::aggregation::Kind const> agg_kinds,
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
                 sparse_column_creator{stream});
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
  host_span<cudf::aggregation::Kind const> agg_kinds,
  bool direct_aggregations,
  global_set_t const& global_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template cudf::table create_sparse_results_table<nullable_global_set_t>(
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  host_span<cudf::aggregation::Kind const> agg_kinds,
  bool direct_aggregations,
  nullable_global_set_t const& global_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
