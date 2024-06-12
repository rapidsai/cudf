/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>

namespace cudf {
namespace detail {
std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  size_type num_rows = source_table.num_rows();
  auto elements      = make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_rows] __device__(auto i) {
      return num_rows - i - 1;
    }));
  auto elements_end = elements + source_table.num_rows();

  return gather(source_table, elements, elements_end, out_of_bounds_policy::DONT_CHECK, stream, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  return std::move(
    cudf::detail::reverse(table_view({source_column}), stream, mr)->release().front());
}
}  // namespace detail

std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_table, stream, mr);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_column, stream, mr);
}
}  // namespace cudf
