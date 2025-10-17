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

#include "helpers.cuh"
#include "output_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace cudf::groupby::detail::hash {
namespace {

/**
 * @brief Functor to create the result columns for hash-based groupby aggregations
 *
 * This functor handles the creation of appropriately typed and sized columns for each
 * aggregation, including special handling for SUM_WITH_OVERFLOW which requires a struct column.
 */
struct result_column_creator {
  size_type output_size;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  explicit result_column_creator(size_type output_size_,
                                 rmm::cuda_stream_view stream_,
                                 rmm::device_async_resource_ref mr_)
    : output_size{output_size_}, stream{stream_}, mr{mr_}
  {
  }

  std::unique_ptr<column> operator()(column_view const& col, aggregation::Kind const& agg) const
  {
    auto const nullable =
      agg != aggregation::COUNT_VALID && agg != aggregation::COUNT_ALL && col.has_nulls();

    // Special handling for SUM_WITH_OVERFLOW which needs a struct column.
    if (agg != aggregation::SUM_WITH_OVERFLOW) {
      auto const col_type =
        is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
      auto const mask_flag = nullable ? mask_state::ALL_NULL : mask_state::UNALLOCATED;
      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), output_size, mask_flag, stream, mr);
    }

    auto const make_empty_column = [&](type_id type_id, size_type size, mask_state mask_state) {
      return make_fixed_width_column(data_type{type_id}, size, mask_state, stream, mr);
    };

    auto make_children = [&make_empty_column](size_type size) {
      std::vector<std::unique_ptr<column>> children;
      // Create sum child column (int64_t) - no null mask needed, struct-level mask handles
      // nullability
      children.push_back(make_empty_column(type_id::INT64, size, mask_state::UNALLOCATED));
      // Create overflow child column (bool) - no null mask needed, only value matters
      children.push_back(make_empty_column(type_id::BOOL8, size, mask_state::UNALLOCATED));
      return children;
    };

    auto [null_mask, null_count] = [&]() -> std::pair<rmm::device_buffer, size_type> {
      if (output_size > 0 && nullable) {
        return {create_null_mask(output_size, mask_state::ALL_NULL, stream, mr), output_size};
      }
      return {rmm::device_buffer{}, 0};
    }();
    return create_structs_hierarchy(
      output_size, make_children(output_size), null_count, std::move(null_mask), stream);
  }
};

}  // anonymous namespace

std::unique_ptr<table> create_results_table(size_type output_size,
                                            table_view const& values,
                                            host_span<aggregation::Kind const> agg_kinds,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> output_cols;
  std::transform(values.begin(),
                 values.end(),
                 agg_kinds.begin(),
                 std::back_inserter(output_cols),
                 result_column_creator{output_size, stream, mr});
  auto result_table = std::make_unique<table>(std::move(output_cols));
  cudf::detail::initialize_with_identity(result_table->mutable_view(), agg_kinds, stream);
  return result_table;
}

template <typename SetType>
rmm::device_uvector<size_type> extract_populated_keys(SetType const& key_set,
                                                      size_type num_total_keys,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<size_type> unique_key_indices(num_total_keys, stream, mr);
  auto const keys_end = key_set.retrieve_all(unique_key_indices.begin(), stream.value());
  unique_key_indices.resize(std::distance(unique_key_indices.begin(), keys_end), stream);
  return unique_key_indices;
}

template rmm::device_uvector<size_type> extract_populated_keys<global_set_t>(
  global_set_t const& key_set,
  size_type num_total_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> extract_populated_keys<nullable_global_set_t>(
  nullable_global_set_t const& key_set,
  size_type num_total_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

rmm::device_uvector<size_type> compute_key_transform_map(
  size_type num_total_keys,
  device_span<size_type const> unique_key_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Map from old key indices (index of the keys in the original input keys table) to new key
  // indices (indices of the keys in the final output table, which contains only the extracted
  // unique keys). Only these extracted unique keys are mapped.
  rmm::device_uvector<size_type> key_transform_map(num_total_keys, stream, mr);
  thrust::scatter(rmm::exec_policy_nosync(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(static_cast<size_type>(unique_key_indices.size())),
                  unique_key_indices.begin(),
                  key_transform_map.begin());

  return key_transform_map;
}

rmm::device_uvector<size_type> compute_target_indices(device_span<size_type const> input,
                                                      device_span<size_type const> transform_map,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<size_type> target_indices(input.size(), stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    input.begin(),
                    input.end(),
                    target_indices.begin(),
                    [new_indices = transform_map.begin()] __device__(size_type const idx) {
                      return idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL ? idx : new_indices[idx];
                    });
  return target_indices;
}

void finalize_output(table_view const& values,
                     std::vector<std::unique_ptr<aggregation>> const& aggregations,
                     std::unique_ptr<table>& agg_results,
                     cudf::detail::result_cache* cache,
                     rmm::cuda_stream_view stream)
{
  auto result_cols = agg_results->release();
  for (size_t i = 0; i < aggregations.size(); i++) {
    auto& result = result_cols[i];
    if (result->nullable()) {
      // Call `null_count` triggers a stream sync for each output column.
      // This needs to be improved by a batch processing kernel, which is requested
      // in https://github.com/rapidsai/cudf/issues/19878.
      result->set_null_count(
        cudf::null_count(result->view().null_mask(), 0, result->size(), stream));
    }
    cache->add_result(values.column(i), *aggregations[i], std::move(result));
  }
  agg_results.reset();  // to make sure any subsequent use will trigger exception
}

}  // namespace cudf::groupby::detail::hash
