/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reverse.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>

#include <stdio.h>

namespace cudf {
namespace detail {
std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  thrust::counting_iterator<cudf::size_type> iter(0);
  // rmm::device_uvector<cudf::size_type> indices(source_table.num_rows(), stream);
  thrust::device_vector<cudf::size_type> indices(source_table.num_rows());
  thrust::copy(iter, iter + source_table.num_rows(), indices.begin());

  thrust::reverse(rmm::exec_policy(stream), indices.begin(), indices.end());

  return gather(
    source_table, indices.begin(), indices.end(), out_of_bounds_policy::DONT_CHECK, stream, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  // rmm::device_uvector<size_type> indices(output_size, stream);
  return std::unique_ptr<column>{};
}
}  // namespace detail

std::unique_ptr<table> reverse(table_view const& source_table, rmm::mr::device_memory_resource* mr)
{
  return detail::reverse(source_table, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::mr::device_memory_resource* mr)
{
  return detail::reverse(source_column, rmm::cuda_stream_default, mr);
}
}  // namespace cudf