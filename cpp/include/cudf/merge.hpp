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

#include <vector>
#include <cudf/cudf.h>
#include <cudf/types.hpp>

#include <cudf/table/row_operators.cuh>


namespace { // anonym.
/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };

/**
 * @brief Tagged index type: `thrust::get<0>` indicates left/right side, 
 * `thrust::get<1>` indicates the row index
 */  
using index_type = thrust::tuple<side, cudf::size_type>;  
} //anonym.

namespace cudf {
namespace experimental {
/**
 * @brief tagged_element_relational_comparator uses element_relational_comparator to provide "tagged-index" comparation logic.
 *
 * Special treatment is necessary in several thrust algorithms (e.g., merge()) where 
 * the index affinity to the side is not guaranteed; i.e., the algorithms rely on 
 * binary functors (predicates) where the operands may transparently switch sides.
 *
 * For example, 
 *         thrust::merge(left_container,
 *                       right_container,
 *                       predicate(lhs, rhs){...});
 *         can create 4 different use-cases, inside predicate(...):
 *
 *         1. lhs refers to the left container; rhs to the right container;
 *         2. vice-versa;
 *         3. both lhs and rhs actually refer to the left container;
 *         4. both lhs and rhs actually refer to the right container;
 *
 * Because of that, one cannot rely on the predicate having *fixed* references to the containers.
 * Each invocation may land in a different situation (among the 4 above) than any other invocation.
 * Also, one cannot just manipulate lhs, rhs (indices) alone; because, if predicate always applies
 * one index to one container and the other index to the other container, 
 * switching the indices alone won't suffice in the cases (3) or (4), 
 * where the also the containers must be changed (to just one instead of two) 
 * independently of indices;
 *
 * As a result, a special comparison logic is necessary whereby the index is "tagged" with side information
 * and consequently comparator functors (predicates) must operate 
 * on these tagged indices rather than on raw indices.
 *  
 */
template <bool has_nulls = true>
struct tagged_element_relational_comparator {

  __host__ __device__
  tagged_element_relational_comparator(column_device_view lhs,
                                       column_device_view rhs,
                                       null_order null_precedence)
    : lhs{lhs}, rhs{rhs}, null_precedence{null_precedence}
  {
  }


  template <typename Element,
            std::enable_if_t<cudf::is_relationally_comparable<
                               Element, Element>()>* = nullptr>
  __device__
  weak_ordering operator()(index_type lhs_tagged_index,
                           index_type rhs_tagged_index) const noexcept {

    side l_side = thrust::get<0>(lhs_tagged_index);
    side r_side = thrust::get<0>(rhs_tagged_index);

    cudf::size_type l_indx = thrust::get<1>(lhs_tagged_index);
    cudf::size_type r_indx = thrust::get<1>(rhs_tagged_index);

    column_device_view const* ptr_left_dview{l_side == side::LEFT ? &lhs : &rhs };
    
    column_device_view const* ptr_right_dview{r_side == side::LEFT ? &lhs : &rhs };
 
    auto erl_comparator = element_relational_comparator<has_nulls>(*ptr_left_dview, *ptr_right_dview, null_precedence);

    return cudf::experimental::type_dispatcher(lhs.type(),
                                               erl_comparator,
                                               l_indx, r_indx);
    
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_relationally_comparable<
                Element, Element>()>* = nullptr>
  __device__
  weak_ordering operator()(index_type lhs_tagged_index,
                           index_type rhs_tagged_index) const noexcept {
    release_assert(false &&
                   "Attempted to compare elements of uncomparable types.");
  }
private:
  column_device_view lhs;
  column_device_view rhs;
  null_order null_precedence;
};

/**
 * @brief The equivalent of `row_lexicographic_comparator` for tagged indices.
 */
template <bool has_nulls = true>
struct row_lexicographic_tagged_comparator { 
  row_lexicographic_tagged_comparator(table_device_view lhs, table_device_view rhs,
                                      order const* column_order = nullptr,
                                      null_order const* null_precedence = nullptr)
      : _lhs{lhs},
        _rhs{rhs},
        _column_order{column_order},
        _null_precedence{null_precedence} {
    // Add check for types to be the same.
    CUDF_EXPECTS(_lhs.num_columns() == _rhs.num_columns(),
                 "Mismatched number of columns.");
  }

  __device__
  bool operator()(index_type lhs_tagged_index,
                  index_type rhs_tagged_index) const noexcept {
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      bool ascending =
          (_column_order == nullptr) or (_column_order[i] == order::ASCENDING);

      weak_ordering state{weak_ordering::EQUIVALENT};
      null_order null_precedence = _null_precedence == nullptr ?
                                     null_order::BEFORE: _null_precedence[i];

      auto comparator = tagged_element_relational_comparator<has_nulls>{
          _lhs.column(i), _rhs.column(i), null_precedence};

      state = cudf::experimental::type_dispatcher(_lhs.column(i).type(), comparator,
                                         lhs_tagged_index, rhs_tagged_index);

      if (state == weak_ordering::EQUIVALENT) {
        continue;
      }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  null_order const*  _null_precedence{};
  order const* _column_order{};
};


/**
 * @brief Merge sorted tables.
 * 
 * Merges two sorted tables into one sorted table
 * containing data from both tables.
 *
 * Example 1:
 * input:
 * table 1 => col 1 {0, 1, 2, 3}
 *            col 2 {4, 5, 6, 7}
 * table 2 => col 1 {1, 2}
 *            col 2 {8, 9}
 * output:
 * table => col 1 {0, 1, 1, 2, 2, 3}
 *          col 2 {4, 5, 8, 6, 9, 7}
 *
 * Example 2: 
 * input:
 * table 1 => col 0 {1, 0}
 *            col 1 {'c', 'b'}
 *            col 2 {RED, GREEN}
 *
 *
 * table 2 => col 0 {1}
 *            col 1 {'a'}
 *            col 2 {NULL}
 *
 *  with key_cols[] = {0,1}
 *  and  asc_desc[] = {ASC, ASC};
 *
 *  Lex-sorting is on columns {0,1}; hence, lex-sorting of ((L0 x L1) V (R0 x R1)) is:
 *  (0,'b', GREEN), (1,'a', NULL), (1,'c', RED)
 *
 *  (third column, the "color", just "goes along for the ride"; 
 *   meaning is permutted according to the data movements dictated 
 *   by lexicographic ordering of columns 0 and 1);
 *
 *   with result columns:
 *
 *   Res0 = {0,1,1}
 *   Res1 = {'b', 'a', 'c'}
 *   Res2 = {GREEN, NULL, RED}
 *
 * @Param[in] left_table A sorted table to be merged
 * @Param[in] right_table A sorted table to be merged
 * @Param[in] key_cols Indices of left_cols and right_cols to be used
 *                     for comparison criteria
 * @Param[in] column_order Sort order types of columns indexed by key_cols
 * @Param[in] null_precedence Array indicating the order of nulls with respect to non-nulls for the indexing columns (key_cols)
 *
 * @Returns A table containing sorted data from left_table and right_table
 */
std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                                 table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& column_order,
                                                 std::vector<cudf::null_order> const& null_precedence = {},
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf

