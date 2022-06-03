/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/lists/set_operations.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf::lists {
namespace detail {

namespace {

/**
 * @brief Create a hash map with keys are indices of all elements in the input column.
 */
auto create_map(column_view const& input, rmm::cuda_stream_view stream)
{
  //
}

/**
 * @brief Check the existence of rows in the keys column in the hash map.
 */
auto check_contains(map_type const& map, column_view const& keys)
{
  //
}

/**
 * @brief Generate labels for elements in the child column of the input lists column.
 * @param input
 */
auto generate_labels(lists_column_view const& input)
{
  //
}

/**
 * @brief Extract rows from the input table based on the boolean values in the input `condition`
 * column.
 */
auto extract_if()

{
  //
}

/**
 * @brief Reconstruct an offsets column from the input labels array.
 */
auto reconstruct_offsets()
{
  //
}

}  // namespace

std::unique_ptr<column> overlap(lists_column_view const& lhs,
                                    lists_column_view const& rhs,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  // - Insert lhs child elements into map.
  // - Check contains for rhs child elements.
  // - Generate labels for rhs child elements.
  // - `reduce_by_key` with `logical_or` functor and keys are labels, values are contains.
  return nullptr;
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // - Insert lhs child elements into map.
  // - Check contains for rhs child element.
  // - Generate labels for rhs child elements.
  // - copy_if {indices, labels} for rhs child elements using contains conditions to {gather_map,
  //   intersect_labels}.
  // - output_child = pull rhs child elements from gather_map.
  // - output_offsets = reconstruct offsets from intersect_labels.
  // - return lists_column(output_child, output_offsets)
  return nullptr;
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  // - concatenate_row lhs and set_except(rhs, lhs)
  return nullptr;
}

std::unique_ptr<column> set_except(lists_column_view const& lhs,
                                   lists_column_view const& rhs,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  // - Insert rhs child elements.
  // - Check contains for lhs child element.
  // - Invert contains for lhs child element.
  // - Generate labels for lhs child elements.
  // - copy_if {indices, labels} using the inverted contains conditions to {gather_map,
  //   except_labels} for lhs child elements.
  // - Pull lhs child elements from gather_map.
  // - Reconstruct output offsets from except_labels for lhs.
  return nullptr;
}

}  // namespace detail

std::unique_ptr<column> overlap(lists_column_view const& lhs,
                                lists_column_view const& rhs,
                                rmm::mr::device_memory_resource* mr)
{
  return detail::overlap(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_intersect(lists_column_view const& lhs,
                                      lists_column_view const& rhs,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::set_intersect(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_union(lists_column_view const& lhs,
                                  lists_column_view const& rhs,
                                  rmm::mr::device_memory_resource* mr)
{
  return detail::set_union(lhs, rhs, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> set_except(lists_column_view const& lhs,
                                   lists_column_view const& rhs,
                                   rmm::mr::device_memory_resource* mr)
{
  return detail::set_except(lhs, rhs, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
