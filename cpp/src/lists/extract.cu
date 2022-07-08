/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sequence.hpp>
#include <cudf/lists/detail/extract.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/extract.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>

#include <limits>

namespace cudf {
namespace lists {
namespace detail {
namespace {

/**
 * @brief Helper to construct a column of indices, for use with `segmented_gather()`.
 *
 * When indices are specified as a column, e.g. `{5, -4, 3, -2, 1, null}`,
 * the column returned is:                      `{5, -4, 3, -2, 1, MAX_SIZE_TYPE}`.
 * All null indices are replaced with `MAX_SIZE_TYPE = numeric_limits<size_type>::max()`.
 *
 * The returned column can then be used to construct a lists column, for use
 * with `segmented_gather()`.
 */
std::unique_ptr<cudf::column> make_index_child(column_view const& indices,
                                               size_type,
                                               rmm::cuda_stream_view stream)
{
  // New column, near identical to `indices`, except with null values replaced.
  // `segmented_gather()` on a null index should produce a null row.
  if (not indices.nullable()) { return std::make_unique<column>(indices, stream); }

  auto const d_indices = column_device_view::create(indices, stream);
  // Replace null indices with MAX_SIZE_TYPE, so that gather() returns null for them.
  auto const null_replaced_iter_begin =
    cudf::detail::make_null_replacement_iterator(*d_indices, std::numeric_limits<size_type>::max());
  auto index_child =
    make_numeric_column(data_type{type_id::INT32}, indices.size(), mask_state::UNALLOCATED, stream);
  thrust::copy_n(rmm::exec_policy(stream),
                 null_replaced_iter_begin,
                 indices.size(),
                 index_child->mutable_view().begin<size_type>());
  return index_child;
}

/**
 * @brief Helper to construct a column of indices, for use with `segmented_gather()`.
 *
 * When indices are specified as a size_type, e.g. `7`,
 * the column returned is: `{ 7, 7, 7, 7, 7 }`.
 *
 * The returned column can then be used to construct a lists column, for use
 * with `segmented_gather()`.
 */
std::unique_ptr<cudf::column> make_index_child(size_type index,
                                               size_type num_rows,
                                               rmm::cuda_stream_view stream)
{
  auto index_child =  // [index, index, index, ..., index]
    make_numeric_column(data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED, stream);
  thrust::fill_n(
    rmm::exec_policy(stream), index_child->mutable_view().begin<size_type>(), num_rows, index);
  return index_child;
}

/**
 * @brief Helper to construct offsets column for an index vector.
 *
 * Constructs the sequence: `{ 0, 1, 2, 3, ... num_lists + 1}`.
 * This may be used to construct an "index-list" column, where each list row
 * has a single element.
 */
std::unique_ptr<cudf::column> make_index_offsets(size_type num_lists, rmm::cuda_stream_view stream)
{
  return cudf::detail::sequence(
    num_lists + 1, cudf::scalar_type_t<size_type>(0, true, stream), stream);
}

}  // namespace

/**
 * @copydoc cudf::lists::extract_list_element
 * @tparam index_t The type used to specify the index values (either column_view or size_type)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename index_t>
std::unique_ptr<column> extract_list_element_impl(lists_column_view lists_column,
                                                  index_t const& index,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  auto const num_lists = lists_column.size();
  if (num_lists == 0) { return empty_like(lists_column.child()); }

  // Given an index (or indices vector), an index lists column may be constructed,
  // with each list row having a single element.
  // E.g.
  // 1. If index = 7, index_lists_column = { {7}, {7}, {7}, {7}, ... }.
  // 2. If indices = {4, 3, 2, 1, null},
  //    index_lists_column = { {4}, {3}, {2}, {1}, {MAX_SIZE_TYPE} }.

  auto const index_lists_column = make_lists_column(num_lists,
                                                    make_index_offsets(num_lists, stream),
                                                    make_index_child(index, num_lists, stream),
                                                    0,
                                                    {},
                                                    stream);

  auto extracted_lists = segmented_gather(
    lists_column, index_lists_column->view(), out_of_bounds_policy::NULLIFY, stream, mr);

  return std::move(extracted_lists->release().children[lists_column_view::child_column_index]);
}

/**
 * @copydoc cudf::lists::extract_list_element
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type const index,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element_impl(lists_column, index, stream, mr);
}

std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element_impl(lists_column, indices, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view const&,
 *                                            size_type,
 *                                            rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             size_type index,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element_impl(lists_column, index, cudf::default_stream_value, mr);
}

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view const&,
 *                                            column_view const&,
 *                                            rmm::mr::device_memory_resource*)
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             column_view const& indices,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(indices.size() == lists_column.size(),
               "Index column must have as many elements as lists column.");
  return detail::extract_list_element_impl(lists_column, indices, cudf::default_stream_value, mr);
}

}  // namespace lists
}  // namespace cudf
