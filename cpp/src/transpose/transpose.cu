/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transpose.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/reshape.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/transpose.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
std::pair<std::unique_ptr<column>, table_view> transpose(table_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  // If there are no rows in the input, return successfully
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    return std::make_pair(std::make_unique<column>(), table_view{});
  }

  // Check datatype homogeneity
  auto const dtype = input.column(0).type();
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [dtype](auto const& col) { return dtype == col.type(); }),
    "Column type mismatch");

  auto output_column = cudf::interleave_columns(input, mr);
  auto one_iter      = thrust::make_counting_iterator<size_type>(1);
  auto splits_iter   = thrust::make_transform_iterator(
    one_iter, [width = input.num_columns()](size_type idx) { return idx * width; });
  auto splits = std::vector<size_type>(splits_iter, splits_iter + input.num_rows() - 1);
  auto output_column_views = cudf::split(output_column->view(), splits);

  return std::make_pair(std::move(output_column), table_view(output_column_views));
}
}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> transpose(table_view const& input,
                                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::transpose(input, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
