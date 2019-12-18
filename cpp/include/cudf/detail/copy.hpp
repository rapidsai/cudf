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

/**
 * @copydoc cudf::experimental::contiguous_split
 *
 * @param stream Optional CUDA stream on which to execute kernels
 **/
std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                      cudaStream_t stream = 0);

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
std::unique_ptr<column> allocate_like(column_view const& input, size_type size,
                                      mask_allocation_policy mask_alloc = 
                                          mask_allocation_policy::RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource(),
                                      cudaStream_t stream = 0);


/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or 
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following rule:
 *          output[i] = (boolean_mask[i]) ? lhs[i] : rhs[i]
 *         
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length
 * @throws cudf::logic_error if boolean mask is not of type bool8
 * @throws cudf::logic_error if boolean mask is not of the same length as lhs and rhs  
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] column_view representing "left (true) / right (false)" boolean for each element
 * @param[in] mr resource for allocating device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
