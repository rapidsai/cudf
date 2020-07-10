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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace detail {
/**
 * @brief Constructs a zero-copy `column_view`/`mutable_column_view` of the
 * elements in the range `[begin,end)` in `input`.
 *
 * @note It is the caller's responsibility to ensure that the returned view
 * does not outlive the viewed device memory.
 *
 * @throws cudf::logic_error if `begin < 0`, `end < begin` or
 * `end > input.size()`.
 *
 * @param[in] input View of input column to slice
 * @param[in] begin Index of the first desired element in the slice (inclusive).
 * @param[in] end Index of the last desired element in the slice (exclusive).
 *
 * @return ColumnView View of the elements `[begin,end)` from `input`.
 **/
template <typename ColumnView>
ColumnView slice(ColumnView const& input, cudf::size_type begin, cudf::size_type end)
{
  static_assert(std::is_same<ColumnView, cudf::column_view>::value or
                  std::is_same<ColumnView, cudf::mutable_column_view>::value,
                "slice can be performed only on column_view and mutable_column_view");
  CUDF_EXPECTS(begin >= 0, "Invalid beginning of range.");
  CUDF_EXPECTS(end >= begin, "Invalid end of range.");
  CUDF_EXPECTS(end <= input.size(), "Slice range out of bounds.");

  std::vector<ColumnView> children{};
  children.reserve(input.num_children());
  for (size_type index = 0; index < input.num_children(); index++) {
    children.emplace_back(input.child(index));
  }

  return ColumnView(input.type(),
                    end - begin,
                    input.head(),
                    input.null_mask(),
                    cudf::UNKNOWN_NULL_COUNT,
                    input.offset() + begin,
                    children);
}

/**
 * @copydoc cudf::slice(column_view const&,std::vector<size_type> const&)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<column_view> slice(column_view const& input,
                               std::vector<size_type> const& indices,
                               cudaStream_t stream = 0);

/**
 * @copydoc cudf::contiguous_split
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::vector<contiguous_split_result> contiguous_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::allocate_like(column_view const&, size_type, mask_allocation_policy,
 * rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> allocate_like(
  column_view const& input,
  size_type size,
  mask_allocation_policy mask_alloc   = mask_allocation_policy::RETAIN,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::copy_if_else( column_view const&, column_view const&,
 * column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::copy_if_else( scalar const&, column_view const&,
 * column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::copy_if_else( column_view const&, scalar const&,
 * column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::copy_if_else( scalar const&, scalar const&,
 * column_view const&, rmm::mr::device_memory_resource*)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::sample
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement = sample_with_replacement::FALSE,
                              int64_t const seed                  = 0,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
