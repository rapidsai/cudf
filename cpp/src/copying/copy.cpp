/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>

namespace cudf {
namespace detail {
namespace {

inline mask_state should_allocate_mask(mask_allocation_policy mask_alloc, bool mask_exists)
{
  if ((mask_alloc == mask_allocation_policy::ALWAYS) ||
      (mask_alloc == mask_allocation_policy::RETAIN && mask_exists)) {
    return mask_state::UNINITIALIZED;
  } else {
    return mask_state::UNALLOCATED;
  }
}

}  // namespace

/*
 * Creates an uninitialized new column of the specified size and same type as
 * the `input`. Supports only fixed-width types.
 */
std::unique_ptr<column> allocate_like(column_view const& input,
                                      size_type size,
                                      mask_allocation_policy mask_alloc,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(is_fixed_width(input.type()), "Expects only fixed-width type column");
  mask_state allocate_mask = should_allocate_mask(mask_alloc, input.nullable());

  auto op = [&](auto const& child) { return allocate_like(child, size, mask_alloc, stream, mr); };
  auto begin = thrust::make_transform_iterator(input.child_begin(), op);
  std::vector<std::unique_ptr<column>> children(begin, begin + input.num_children());

  return std::make_unique<column>(input.type(),
                                  size,
                                  rmm::device_buffer(size * size_of(input.type()), stream, mr),
                                  detail::create_null_mask(size, allocate_mask, stream, mr),
                                  state_null_count(allocate_mask, input.size()),
                                  std::move(children));
}

}  // namespace detail

/*
 * Initializes and returns an empty column of the same type as the `input`.
 */
std::unique_ptr<column> empty_like(column_view const& input)
{
  CUDF_FUNC_RANGE();

  std::vector<std::unique_ptr<column>> children;
  std::transform(input.child_begin(),
                 input.child_end(),
                 std::back_inserter(children),
                 [](column_view const& col) { return empty_like(col); });

  return std::make_unique<cudf::column>(
    input.type(), 0, rmm::device_buffer{}, rmm::device_buffer{}, 0, std::move(children));
}

/*
 * Creates a table of empty columns with the same types as the `input_table`
 */
std::unique_ptr<table> empty_like(table_view const& input_table)
{
  CUDF_FUNC_RANGE();
  std::vector<std::unique_ptr<column>> columns(input_table.num_columns());
  std::transform(input_table.begin(), input_table.end(), columns.begin(), [&](column_view in_col) {
    return empty_like(in_col);
  });
  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> allocate_like(column_view const& input,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::allocate_like(input, input.size(), mask_alloc, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> allocate_like(column_view const& input,
                                      size_type size,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::allocate_like(input, size, mask_alloc, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
