/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>

#include <thrust/count.h>

namespace cudf {
namespace detail {

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(gather_map.has_nulls() == false, "gather_map contains nulls");

  // create index type normalizing iterator for the gather_map
  auto map_begin = indexalator_factory::make_input_iterator(gather_map);
  auto map_end   = map_begin + gather_map.size();

  if (bounds == out_of_bounds_policy::FAIL) {
    cudf::size_type begin =
      neg_indices == negative_index_policy::ALLOWED ? -source_table.num_rows() : 0;
    cudf::size_type end = source_table.num_rows();
    CUDF_EXPECTS(gather_map.size() == thrust::count_if(rmm::exec_policy(stream)->on(stream.value()),
                                                       map_begin,
                                                       map_end,
                                                       [begin, end] __device__(size_type index) {
                                                         return ((index >= begin) && (index < end));
                                                       }),
                 "Index out of bounds.");
  }

  if (neg_indices == negative_index_policy::ALLOWED) {
    cudf::size_type n_rows = source_table.num_rows();
    auto idx_converter     = [n_rows] __device__(size_type in) {
      return ((in % n_rows) + n_rows) % n_rows;
    };
    return gather(source_table,
                  thrust::make_transform_iterator(map_begin, idx_converter),
                  thrust::make_transform_iterator(map_end, idx_converter),
                  bounds == out_of_bounds_policy::IGNORE,
                  stream,
                  mr);
  }
  return gather(
    source_table, map_begin, map_end, bounds == out_of_bounds_policy::IGNORE, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              bool check_bounds,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  auto index_policy = is_unsigned(gather_map.type()) ? detail::negative_index_policy::NOT_ALLOWED
                                                     : detail::negative_index_policy::ALLOWED;

  return detail::gather(
    source_table,
    gather_map,
    check_bounds ? detail::out_of_bounds_policy::FAIL : detail::out_of_bounds_policy::NULLIFY,
    index_policy,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cudf
