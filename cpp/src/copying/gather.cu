/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>

namespace cudf {
namespace detail {

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

  // create index type normalizing iterator for the gather_map
  auto map_begin = indexalator_factory::make_input_iterator(gather_map);
  auto map_end   = map_begin + gather_map.size();

  if (neg_indices == negative_index_policy::ALLOWED) {
    cudf::size_type n_rows = source_table.num_rows();
    auto idx_converter     = cuda::proclaim_return_type<size_type>(
      [n_rows] __device__(size_type in) { return in < 0 ? in + n_rows : in; });
    return gather(source_table,
                  thrust::make_transform_iterator(map_begin, idx_converter),
                  thrust::make_transform_iterator(map_end, idx_converter),
                  bounds_policy,
                  stream,
                  mr);
  }
  return gather(source_table, map_begin, map_end, bounds_policy, stream, mr);
}

std::unique_ptr<table> gather(table_view const& source_table,
                              device_span<size_type const> const gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(gather_map.size() <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "gather map size exceeds the column size limit",
               std::overflow_error);
  auto map_col = column_view(data_type{type_to_id<size_type>()},
                             static_cast<size_type>(gather_map.size()),
                             gather_map.data(),
                             nullptr,
                             0);
  return gather(source_table, map_col, bounds_policy, neg_indices, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  auto index_policy = is_unsigned(gather_map.type()) ? detail::negative_index_policy::NOT_ALLOWED
                                                     : detail::negative_index_policy::ALLOWED;

  return detail::gather(source_table, gather_map, bounds_policy, index_policy, stream, mr);
}

}  // namespace cudf
