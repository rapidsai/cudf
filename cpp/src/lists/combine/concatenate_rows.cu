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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/detail/combine.hpp>
#include <cudf/lists/detail/interleave_columns.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
/**
 * @copydoc cudf::lists::concatenate_rows
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.num_columns() > 0, "The input table must have at least one column.");

  auto const entry_type = lists_column_view(*input.begin()).child().type();
  for (auto const& col : input) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(not cudf::is_nested(child_col.type()), "Nested types are not supported.");
    CUDF_EXPECTS(entry_type == child_col.type(),
                 "The types of entries in the input columns must be the same.");
  }

  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (num_rows == 0) { return cudf::empty_like(input.column(0)); }
  if (num_cols == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }

  // Memory resource for temporary data.
  auto const default_mr = rmm::mr::get_current_device_resource();

  // Interleave the input table into one column.
  auto const has_null_mask = std::any_of(
    std::cbegin(input), std::cend(input), [](auto const& col) { return col.nullable(); });
  auto interleaved_columns = detail::interleave_columns(input, has_null_mask, stream, default_mr);

  // Generate a lists column which has child column is the interleaved_columns.
  // The new nested lists column will have each row is a list of `num_cols` list elements.
  static_assert(std::is_same_v<offset_type, int32_t> and std::is_same_v<size_type, int32_t>);
  auto list_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, default_mr);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(num_rows + 1),
                    list_offsets->mutable_view().template begin<offset_type>(),
                    [num_cols] __device__(auto const idx) { return idx * num_cols; });
  auto const nested_lists_col = make_lists_column(num_rows,
                                                  std::move(list_offsets),
                                                  std::move(interleaved_columns),
                                                  0,
                                                  rmm::device_buffer{},
                                                  stream,
                                                  default_mr);

  // Concatenate lists on each row of the nested lists column, producing the desired output.
  return concatenate_list_elements(nested_lists_col->view(), null_policy, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_rows
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_rows(input, null_policy, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
