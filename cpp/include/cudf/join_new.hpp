/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {

namespace join {

  /**
   * @brief Type defining join comparison operators
   *
   * Indicates how two elements `a` and `b` compare with one and another.
   */
  enum class join_comparison_operator {
    LESS_THAN,             ///< Compare a < b
    LESS_THAN_OR_EQUAL,    ///< Compare a <= b
    EQUAL,                 ///< Compare a == b
    NOT_EQUAL,             ///< Compare a != b
    GREATER_THAN,          ///< Compare a >  b
    GREATER_THAN_OR_EQUAL, ///< Compare a >= b
  };

  /**
   * Indicates implementation of join
   */
  enum class join_implementation {
    NESTED_LOOP,          ///< Iterate over all possible outputs
    SORT_MERGE,           ///< Sort rows and merge to identify join pairs
    HASH                  ///< Use hash table to identify join pairs
  };

  /**
   * @brief Type defining join operations
   */
  struct join_operation {
    join_comparison_operator op;                   ///< type of comparison
    cudf::size_type          left_column_idx;      ///< index of left column to compare
    cudf::size_type          right_column_idx;     ///< index of right column to compare
  };
  
  // NOTES:
  //    1) primary_join_ops and secondary_join_ops are a collection
  //       of join operations connected by AND.
  //    2) For sort/merge and has joins, if secondary_join_ops
  //       is empty we skip that step (making the join behavior the same
  //       as the current behavior)
  //    3) left/right columns output identifies specific columns (and order)
  //       for output.  All left columns would before all right columns specified.
  //       Similar to columns_in_common, but offers a bit more control
  //       *** REVERTED TO columns_in_common, this change can be considered
  //       separately.  I worked through the code for constructing the output and
  //       now understand why columns_in_common might be slightly better in the case
  //       of a full join.  But I do wonder why we need this complexity.  And if we do,
  //       couldn't we just figure out what columns_in_common should be from the
  //       join parameters?
  //    4) This interface would support inequality joins, although we could first
  //       implement only equijoins.
  //       *** eventually if we need to support complex and/or/not expressions
  //       the primary and secondary join ops could be changed to expression trees
  //       (although maybe only the secondary join)
  //    5) Would want to promote join_kind out of experimental::detail
  //

//
//  Option 1: namespace by type of join
//
namespace inner_join {
  template <typename Filter>
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace inner_join

namespace left_join {
  template <typename Filter>
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace left_join

namespace full_join {
  template <typename Filter>
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                          cudf::table_view const& right,
                                          std::vector<join_operation> const& primary_join_ops,
                                          std::vector<join_operation> const& secondary_join_ops,
                                          std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace full_join

//
// Option 2:  namespace by implementation
//
namespace nested_loop {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace nested_loop

namespace sort_merge {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace sort_merge
  
namespace hash {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace hash
  
} //namespace join

} //namespace cudf
