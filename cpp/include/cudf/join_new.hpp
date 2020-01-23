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

struct IdentityFilter {
  bool __device__ operator()(table_device_view const& left, table_device_view const& right,
                             cudf::size_type left_idx, cudf::size_type right_idx) {
    return true;
  }
};

class EqualColumnFilter {
public:
  EqualColumnFilter(cudf::size_type const* left_on,
                    cudf::size_type const* right_on,
                    cudf::size_type size): left_on_(left_on), right_on_(right_on), size_(size) {}
  
  bool __device__ operator()(table_device_view const& left, table_device_view const& right,
                             cudf::size_type left_idx, cudf::size_type right_idx) {

    bool reply = true;
    for (cudf::size_type i = 0 ; reply && (i < size_) ; ++i) {
      auto l = left.column(left_on_[i]);
      auto r = right.column(right_on_[i]);

      // TODO: need to work has_nulls and nulls_are_equal correctly in the below call
      reply = cudf::experimental::type_dispatcher(l.type(),
                                                  element_equality_comparator<true>(l, r, false),
                                                  left_idx, right_idx);
    }

    return reply;
  }

private:
  cudf::size_type const* left_on_;
  cudf::size_type const* right_on_;
  cudf::size_type        size_;
};

//
//  Option 1: namespace by type of join
//
namespace inner_join {
  template <typename Filter>
  std::unique_ptr<table> nested_loop(cudf::table_view const& left,
                                     cudf::table_view const& right,
                                     std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<table> sort_merge(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    std::vector<cudf::size_type> const& base_left_on,
                                    std::vector<cudf::size_type> const& base_right_on,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<table> hash(cudf::table_view const& left,
                            cudf::table_view const& right,
                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                            std::vector<cudf::size_type> const& base_left_on,
                            std::vector<cudf::size_type> const& base_right_on,
                            Filter filter,
                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace inner_join

namespace left_join {
  template <typename Filter>
  std::unique_ptr<table> nested_loop(cudf::table_view const& left,
                                     cudf::table_view const& right,
                                     std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<table> sort_merge(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    std::vector<cudf::size_type> const& base_left_on,
                                    std::vector<cudf::size_type> const& base_right_on,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<table> hash(cudf::table_view const& left,
                            cudf::table_view const& right,
                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                            std::vector<cudf::size_type> const& base_left_on,
                            std::vector<cudf::size_type> const& base_right_on,
                            Filter filter,
                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace left_join

namespace full_join {
  template <typename Filter>
  std::unique_ptr<table> nested_loop(cudf::table_view const& left,
                                     cudf::table_view const& right,
                                     std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                     Filter filter,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  /**
   * base_left_on and base_right_on provide the base join clause(s) which create an intermediate
   * result.  Then we apply Filter to reduce the intermediate result to a final join result which
   * gets gathered into the new table.
   *
   * nested_loop_join above has no base join, we only apply the filter.
   */
  template <typename Filter>
  std::unique_ptr<table> sort_merge(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    std::vector<cudf::size_type> const& base_left_on,
                                    std::vector<cudf::size_type> const& base_right_on,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

template <typename Filter>
std::unique_ptr<table> hash(cudf::table_view const& left,
                            cudf::table_view const& right,
                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                            std::vector<cudf::size_type> const& base_left_on,
                            std::vector<cudf::size_type> const& base_right_on,
                            Filter filter,
                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace full_join


//
// Option 2:  namespace by implementation
//
namespace nested_loop {
  template <typename Filter>
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  template <typename Filter>
  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  template <typename Filter>
  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace nested_loop

namespace sort_merge {
  template <typename Filter>
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    std::vector<cudf::size_type> const& base_left_on,
                                    std::vector<cudf::size_type> const& base_right_on,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  template <typename Filter>

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   std::vector<cudf::size_type> const& base_left_on,
                                   std::vector<cudf::size_type> const& base_right_on,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  template <typename Filter>
  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   std::vector<cudf::size_type> const& base_left_on,
                                   std::vector<cudf::size_type> const& base_right_on,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace sort_merge

namespace hash {
  template <typename Filter>
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    std::vector<cudf::size_type> const& base_left_on,
                                    std::vector<cudf::size_type> const& base_right_on,
                                    Filter filter,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  template <typename Filter>
  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   std::vector<cudf::size_type> const& base_left_on,
                                   std::vector<cudf::size_type> const& base_right_on,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  template <typename Filter>
  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   std::vector<cudf::size_type> const& base_left_on,
                                   std::vector<cudf::size_type> const& base_right_on,
                                   Filter filter,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace hash

//
//  Option 3:  Something completely different
//
namespace completely_different {

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
  
  //
  // NOTES:
  //    1) primary_join_ops and secondary_join_ops are a collection
  //       of join operations (as above) connected by AND.
  //    2) For nested loop join, primary_join_ops should be empty
  //    3) For sort/merge and has joins, if secondary_join_ops
  //       is empty we skip that step
  //    4) left/right columns output identifies specific columns (and order)
  //       for output.  All left columns would before all right columns specified.
  //       Similar to columns_in_common, but offers a bit more control
  //       *** REVERTED TO columns_in_common, this change can be considered
  //       separately.  I worked through the code for constructing the output and
  //       now understand why columns_in_common might be slightly better in the case
  //       of a full join.  But I do wonder why we need this complexity.  And if we do,
  //       couldn't we just figure out what columns_in_common should be from the
  //       join parameters?
  //    5) This interface would support inequality joins, although we could first
  //       implement only equijoins.
  //       *** eventually if we need to support complex and/or/not expressions
  //       the primary and secondary join ops could be changed to expression trees
  //       (although maybe only the secondary join)
  //    6) Could combine some of these ideas with option 1 or option 2 if we
  //       like that better.  For example, we could keep the namespace and join
  //       name (providing 9 outer functions) but make the signatures more like this.
  //    7) Would want to promote join_kind out of experimental::detail
  //

  //
  //  So the following three calls would be equivalent (across these three options):
  //
  //  columns_in_common:  { {0, 0} }
  //  base_left_on: { 0 }
  //  base_right_on: { 0 }
  //  primary_join_ops: { {EQUAL, 0, 0 } }
  //  secondary_join_ops: {}
  //
  //  # Option 1
  //  cudf::join::inner_join::hash(left, right, columns_in_common, base_left_on, base_right_on, IdentityFilter{})
  //
  //  # Option 2
  //  cudf::join::hash::inner_join(left, right, columns_in_common, base_left_on, base_right_on, IdentityFilter{})
  //
  //  # Option 3
  //  cudf::join::completely_different::join(join_kind::INNER_JOIN, join_implementation::HASH,
  //                                         left, right, primary_join_ops, secondary_join_ops,
  //                                         columns_in_common);
  //
  //
  std::unique_ptr<table> join(cudf::experimental::detail::join_kind kind,
                              join_implementation impl,
                              cudf::table_view const& left,
                              cudf::table_view const& right,
                              std::vector<join_operation> const& primary_join_ops,
                              std::vector<join_operation> const& secondary_join_ops,
                              std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

namespace nested_loop {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace nested_loop

namespace sort_merge {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& primary_join_ops,
                                    std::vector<join_operation> const& secondary_join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace sort_merge
  
namespace hash {
  std::unique_ptr<table> inner_join(cudf::table_view const& left,
                                    cudf::table_view const& right,
                                    std::vector<join_operation> const& primary_join_ops,
                                    std::vector<join_operation> const& secondary_join_ops,
                                    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> left_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

  std::unique_ptr<table> full_join(cudf::table_view const& left,
                                   cudf::table_view const& right,
                                   std::vector<join_operation> const& primary_join_ops,
                                   std::vector<join_operation> const& secondary_join_ops,
                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
} //namespace hash
  
} //namespace completely_different

} //namespace join

} //namespace cudf
