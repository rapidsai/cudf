/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#pragma once 

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/copying.hpp>

namespace cudf {
namespace experimental {
namespace detail {
/**---------------------------------------------------------------------------*
 * @brief Constructs a zero-copy `column_view`/`mutable_column_view` of the
 * elements in the range `[begin,end)` in `input`.
 *
 * @note It is the caller's responsibility to ensure that the returned view
 * does not outlive the viewed device memory.
 *
 * @throws `cudf::logic_error` if `begin < 0`, `end < begin` or
 * `end > input.size()`.
 *
 * @param input View of input column to slice
 * @param begin Index of the first desired element in the slice (inclusive).
 * @param end Index of the last desired element in the slice (exclusive).
 *
 * @return ColumnView View of the elements `[begin,end)` from `input`.
 *---------------------------------------------------------------------------**/
template <typename ColumnView>
ColumnView slice(ColumnView const& input,
                  cudf::size_type begin,
                  cudf::size_type end) {
   static_assert(std::is_same<ColumnView, cudf::column_view>::value or
                    std::is_same<ColumnView, cudf::mutable_column_view>::value,
                "slice can be performed only on column_view and mutable_column_view");
   CUDF_EXPECTS(begin >= 0, "Invalid beginning of range.");
   CUDF_EXPECTS(end >= begin, "Invalid end of range.");
   CUDF_EXPECTS(end <= input.size(), "Slice range out of bounds.");

   std::vector<ColumnView> children {};
   children.reserve(input.num_children());
   for (size_type index = 0; index < input.num_children(); index++) {
       children.emplace_back(input.child(index));
   }

   return ColumnView(input.type(), end - begin,
                     input.head(), input.null_mask(),
                     cudf::UNKNOWN_NULL_COUNT,
                     input.offset() + begin, children);
}

/*
 * Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<column> An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view input, cudaStream_t stream = 0);

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] size The desired number of elements that the new column should have capacity for
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the specified number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input, size_type size,
                                      mask_allocation_policy mask_alloc = 
                                          mask_allocation_policy::RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource(),
                                      cudaStream_t stream = 0);

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param[in] input_table Immutable view of input table to emulate
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<table> A table of empty columns with the same types as the columns in `input_table`
 */
std::unique_ptr<table> empty_like(table_view input_table, cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
