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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either experimentalress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/column/column.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>
#include <cudf/cudf.h>
#include <cudf/table/table.hpp>
#include <nvstrings/NVCategory.h>

#include <cuda_runtime.h>
#include <algorithm>

namespace cudf
{
namespace experimental
{
namespace detail
{

inline mask_state should_allocate_mask(mask_allocation_policy mask_alloc, bool mask_exists) {
  if ((mask_alloc == ALWAYS) || (mask_alloc == RETAIN && mask_exists)) {
    return UNINITIALIZED;
  } else {
    return UNALLOCATED;
  }
}

/*
 * Initializes and returns column of the same type as the input.
 */
std::unique_ptr<column> empty_like(column_view input, cudaStream_t stream)
{
  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(empty_like(input.child(index), stream));
  }

  return std::make_unique<column>(input.type(), 0, rmm::device_buffer {},
		                  rmm::device_buffer {}, 0, std::move(children));
}

/*
 * Allocates a new column of the same size and type as the input.
 * Does not copy data.
 */
std::unique_ptr<column> allocate_like(column_view input,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource *mr,
				      cudaStream_t stream)
{
  mask_state allocate_mask = should_allocate_mask(mask_alloc, input.has_nulls());

  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(allocate_like(input.child(index), mask_alloc, mr, stream));
  }

  return std::make_unique<column>(input.type(),
                                  input.size(),
                                  rmm::device_buffer(input.size()*size_of(input.type()), stream, mr),
                                  create_null_mask(input.size(), allocate_mask, stream, mr),
                                  state_null_count(allocate_mask, input.size()),
                                  std::move(children));
}

/*
 * Allocates a new column of specified size of the same type as the input.
 * Does not copy data.
 */
std::unique_ptr<column> allocate_like(column_view input,
		                      gdf_size_type size,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource *mr,
				      cudaStream_t stream)
{
  mask_state allocate_mask = should_allocate_mask(mask_alloc, input.has_nulls());

  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(allocate_like(input.child(index), size, mask_alloc, mr, stream));
  }

  return std::make_unique<column>(input.type(),
                                  size,
                                  rmm::device_buffer(size*size_of(input.type()), stream, mr),
                                  create_null_mask(size, allocate_mask, stream, mr),
                                  state_null_count(allocate_mask, input.size()),
                                  std::move(children));
}

/*
 * Returns an empty table similar to `input_table` with zero sized columns
 */
std::unique_ptr<table> empty_like(table_view input_table, cudaStream_t stream){
  std::vector<std::unique_ptr<column>> columns(input_table.num_columns());
  std::transform(input_table.begin(), input_table.end(), columns.begin(),
    [&](column_view in_col) {
      return empty_like(in_col, stream);
    });

  return  std::make_unique<table>(std::move(columns));
}

/*
 * Allocates a new table of the same size and type of columns
 * Does not copy data.
 */
std::unique_ptr<table> allocate_like(table_view input_table,
                                     mask_allocation_policy mask_alloc,
                                     rmm::mr::device_memory_resource *mr,
				     cudaStream_t stream){
  std::vector<std::unique_ptr<column>> columns(input_table.num_columns());
  std::transform(input_table.begin(), input_table.end(), columns.begin(),
    [&](column_view in_col) {
      return allocate_like(in_col, mask_alloc, mr, stream);
    });

  return std::make_unique<table>(std::move(columns));
}

/*
 * Allocates a new table of the proposed size and same type of columns
 * Does not copy data.
 */
std::unique_ptr<table> allocate_like(table_view input_table,
		                     gdf_size_type size,
                                     mask_allocation_policy mask_alloc,
                                     rmm::mr::device_memory_resource *mr,
				     cudaStream_t stream){
  std::vector<std::unique_ptr<column>> columns(input_table.num_columns());
  std::transform(input_table.begin(), input_table.end(), columns.begin(),
    [&](column_view in_col) {
      return allocate_like(in_col, size, mask_alloc, mr, stream);
    });

  return std::make_unique<table>(std::move(columns));
}
} // namespace detail

std::unique_ptr<column> empty_like(column_view input){
  return detail::empty_like(input, 0);
}

std::unique_ptr<column> allocate_like(column_view input,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource *mr){
  return detail::allocate_like(input, mask_alloc, mr, 0);
}

std::unique_ptr<column> allocate_like(column_view input,
		                      gdf_size_type size,
                                      mask_allocation_policy mask_alloc,
                                      rmm::mr::device_memory_resource *mr){
  return detail::allocate_like(input, size, mask_alloc, mr, 0);
}

std::unique_ptr<table> empty_like(table_view input_table){
  return detail::empty_like(input_table, 0);
}

std::unique_ptr<table> allocate_like(table_view input_table,
                                     mask_allocation_policy mask_alloc,
                                     rmm::mr::device_memory_resource *mr){
  return detail::allocate_like(input_table, mask_alloc, mr, 0);
}

std::unique_ptr<table> allocate_like(table_view input_table,
		                     gdf_size_type size,
                                     mask_allocation_policy mask_alloc,
                                     rmm::mr::device_memory_resource *mr){
  return detail::allocate_like(input_table, size, mask_alloc, mr, 0);
}

} // namespace experimental
} // namespace cudf
