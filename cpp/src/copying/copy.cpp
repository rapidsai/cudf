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
namespace exp
{

/*
 * Initializes and returns column of the same type as the input.
 */
std::unique_ptr<column> empty_like(column_view input)
{
  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(empty_like(input.child(index)));
  }

  return std::make_unique<column>(input.type(), 0, rmm::device_buffer {}, rmm::device_buffer {}, 0, std::move(children));
}
#if 0
/*
 * Allocates a new column of the same size and type as the input.
 * Does not copy data.
 */
column allocate_like(column_view input,
                     mask_state state,
                     cudaStream_t stream,
                     rmm::mr::device_memory_resource *mr)
{
  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(std::make_unique<column>(std::move(allocate_like(input.child(index), state, stream, mr))));
  }

  return column {input.type(),
                 input.size(),
                 rmm::device_buffer(input.size()*size_of(input.type()), stream, mr),
                 create_null_mask(input.size(), state, stream, mr),
                 state_null_count(state, input.size()),
                 std::move(children)};
}

/*
 * Allocates a new column of specified size of the same type as the input.
 * Does not copy data.
 */
column allocate_like(column_view input, gdf_size_type size,
                     mask_state state, cudaStream_t stream,
                     rmm::mr::device_memory_resource *mr)
{
  std::vector<std::unique_ptr<column>> children {};
  children.reserve(input.num_children());
  for (size_type index; index < input.num_children(); index++) {
      children.emplace_back(std::make_unique<column>(std::move(allocate_like(input.child(index), size, state, stream, mr))));
  }

  return column {input.type(),
                 size,
                 rmm::device_buffer(size*size_of(input.type()), stream, mr),
                 create_null_mask(size, state, stream, mr),
                 state_null_count(state, input.size()),
                 std::move(children)};
}

table empty_like(table_view t) {
  std::vector<std::unique_ptr<column>> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [](std::unique_ptr<column> out_col, std::unique_ptr<column> in_col) {
      out_col = std::make_unique<column>(empty_like(*in_col));
      return out_col;
    });

  return table{std::move(columns)};
}

table allocate_like(table_view t,
                    mask_state state,
                    cudaStream_t stream,
                    rmm::mr::device_memory_resource *mr){
  std::vector<std::unique_ptr<column>> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [&](std::unique_ptr<column> out_col, std::unique_ptr<column> in_col) {
      out_col = std::make_unique<column>(allocate_like(*in_col,
			                 state, stream, mr), stream, mr);
      return out_col;
    });

  return table{std::move(columns)};
}

table allocate_like(table_view t,
		    gdf_size_type size,
                    mask_state state,
                    cudaStream_t stream,
                    rmm::mr::device_memory_resource *mr){
  std::vector<std::unique_ptr<column>> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [&](std::unique_ptr<column> out_col, std::unique_ptr<column> in_col) {
      out_col = std::make_unique<column>(allocate_like(*in_col, 
			                  size, state, stream, mr), stream, mr);
      return out_col;
    });

  return table{std::move(columns)};
}

#endif

#if 0
table copy(table_view t) {
  std::vector<std::unique_ptr<column>> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [](std::unique_ptr<column>> out_col, std::unique_ptr<column>> in_col) {
      out_col = std::make_unique<column>(std::move(*c));
      return out_col
    });

  return table{columns};
}

table copy(table_view t, cudaStream_t stream,
           rmm::mr::device_memory_resource* mr) {
  std::vector<std::unique_ptr<column>> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
    [](std::unique_ptr<column>> out_col, std::unique_ptr<column>> in_col) {
      out_col = std::make_unique<column>(*c, stream, mr);
      return out_col
    });

  return table{columns};
}
 #endif
} // namespace ext
} // namespace cudf
