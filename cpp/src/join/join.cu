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
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/gather.cuh>

#include <join/hash_join.cuh>
#include <algorithm>
#include <iterator>

namespace cudf {

namespace experimental {

namespace detail {

bool is_trivial_join(
                     table_view const& left,
                     table_view const& right,
                     std::vector<size_type> const& left_on,
                     std::vector<size_type> const& right_on,
                     join_kind JoinKind) {
  // If there is nothing to join, then send empty table with all columns
  if (left_on.empty() || right_on.empty()) {
      return true;
  }

  // If the inputs are empty, immediately return
  if ((0 == left.num_rows()) && (0 == right.num_rows())) {
      return true;
  }

  // If left join and the left table is empty, return immediately
  if ((join_kind::LEFT_JOIN == JoinKind) && (0 == left.num_rows())) {
      return true;
  }

  // If Inner Join and either table is empty, return immediately
  if ((join_kind::INNER_JOIN == JoinKind) &&
      ((0 == left.num_rows()) || (0 == right.num_rows()))) {
      return true;
  }

  return false;
}


/**---------------------------------------------------------------------------*
 * @brief Returns a vector with non-common indices which is set difference
 * between `[0, num_columns)` and index values in common_column_indices
 *
 * @param num_columns The number of columns , which represents column indices
 * from `[0, num_columns)` in a table
 * @param common_column_indices A vector of common indices which needs to be
 * excluded from `[0, num_columns)`
 * @return vector A vector containing only the indices which are not present in
 * `common_column_indices`
 *---------------------------------------------------------------------------**/
auto non_common_column_indices(
    size_type num_columns,
    std::vector<size_type> const& common_column_indices) {
  CUDF_EXPECTS(common_column_indices.size() <= static_cast<unsigned long>(num_columns),
               "Too many columns in common");
  std::vector<size_type> all_column_indices(num_columns);
  std::iota(std::begin(all_column_indices), std::end(all_column_indices), 0);
  std::vector<size_type> sorted_common_column_indices{
      common_column_indices};
  std::sort(std::begin(sorted_common_column_indices),
            std::end(sorted_common_column_indices));
  std::vector<size_type> non_common_column_indices(num_columns -
                                                common_column_indices.size());
  std::set_difference(std::cbegin(all_column_indices),
                      std::cend(all_column_indices),
                      std::cbegin(sorted_common_column_indices),
                      std::cend(sorted_common_column_indices), std::begin(non_common_column_indices));
   return non_common_column_indices;
}


std::unique_ptr<experimental::table> get_empty_joined_table(
                         table_view const& left,
                         table_view const& right,
                         std::vector<std::pair<size_type, size_type>> const& columns_in_common) {
  std::vector<size_type> right_columns_in_common (columns_in_common.size());
  std::transform(
      columns_in_common.begin(),
      columns_in_common.end(),
      right_columns_in_common.begin(),
      [](auto& col) { return col.second; } );
  std::unique_ptr<experimental::table> empty_left = experimental::empty_like(left);
  std::unique_ptr<experimental::table> empty_right = experimental::empty_like(right);
  std::vector <size_type> right_non_common_indices =
    non_common_column_indices(right.num_columns(), right_columns_in_common);
  table_view tmp_right_table = (*empty_right).select(right_non_common_indices);
  table_view tmp_table{{*empty_left, tmp_right_table}};
  return std::make_unique<experimental::table>(tmp_table);
}


VectorPair
concatenate_vector_pairs(
  VectorPair& a, VectorPair& b)
{
  CUDF_EXPECTS((a.first.size() == a.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  CUDF_EXPECTS((b.first.size() == b.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  if (a.first.size() == 0) {
    return b;
  } else if (b.first.size() == 0) {
    return a;
  }
  auto original_size = a.first.size();
  a.first.resize(a.first.size() + b.first.size());
  a.second.resize(a.second.size() + b.second.size());
  thrust::copy(b.first.begin(), b.first.end(), a.first.begin() + original_size);
  thrust::copy(b.second.begin(), b.second.end(), a.second.begin() + original_size);
  return a;
}


template <typename T>
struct valid_range {
    T start, stop;
    __host__ __device__
    valid_range(
            const T begin,
            const T end) :
        start(begin), stop(end) {}

    __host__ __device__ __forceinline__
    bool operator()(const T index)
    {
        return ((index >= start) && (index < stop));
    }
};


/* --------------------------------------------------------------------------*/
/**
* @brief  Creates a table containing the complement of left join indices.
* This table has two columns. The first one is filled with JoinNoneValue(-1)
* and the second one contains values from 0 to right_table_row_count - 1
* excluding those found in the right_indices column.
*
* @Param right_indices Vector of indices
* @Param left_table_row_count Number of rows of left table
* @Param right_table_row_count Number of rows of right table
* @param stream Optional, stream on which all memory allocations and copies
* will be performed
*
* @Returns  Pair of vectors containing the left join indices complement
*/
/* ----------------------------------------------------------------------------*/
std::pair<rmm::device_vector<size_type>,
rmm::device_vector<size_type>>
get_left_join_indices_complement(
    rmm::device_vector<size_type>& right_indices,
    size_type left_table_row_count,
    size_type right_table_row_count,
    cudaStream_t stream) {

  //Get array of indices that do not appear in right_indices

  //Vector allocated for unmatched result
  rmm::device_vector<size_type> right_indices_complement(right_table_row_count);

  //If left table is empty in a full join call then all rows of the right table
  //should be represented in the joined indices. This is an optimization since
  //if left table is empty and full join is called all the elements in
  //right_indices will be JoinNoneValue, i.e. -1. This if path should
  //produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(
        rmm::exec_policy(stream)->on(stream),
        right_indices_complement.begin(),
        right_indices_complement.end(),
        0);
  } else {
    //Assume all the indices in invalid_index_map are invalid
    rmm::device_vector<size_type> invalid_index_map(right_table_row_count, 1);
    //Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    //invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    //Thus specifying that those locations are valid
    thrust::scatter_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0) + right_indices.size(),
        right_indices.begin(),//Index locations
        right_indices.begin(),//Stencil - Check if index location is valid
        invalid_index_map.begin(),//Output indices
        valid);//Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter = static_cast<size_type>(right_table_row_count);

    //Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(
        rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator(begin_counter),
        thrust::make_counting_iterator(end_counter),
        invalid_index_map.begin(),
        right_indices_complement.begin(),
        thrust::identity<size_type>()) -
      right_indices_complement.begin();
    right_indices_complement.resize(indices_count);
  }

  rmm::device_vector<size_type> left_invalid_indices(
      right_indices_complement.size(), JoinNoneValue);

  return std::make_pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}


/* --------------------------------------------------------------------------*/
/**
 * @brief  Computes the base join operation between two tables and returns the
 * output indices of left and right table as a combined table, i.e. if full
 * join is specified as the join type then left join is called.
 *
 * @throws cudf::logic_error
 * If `left`/`right` table is empty
 * If type mismatch between joining columns
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * @param stream stream on which all memory allocations and copies
 * will be performed
 * @tparam join_kind The type of join to be performed
 *
 * @returns Join output indices vector pair
 */
/* ----------------------------------------------------------------------------*/
template <join_kind JoinKind>
std::pair<rmm::device_vector<size_type>,
  rmm::device_vector<size_type>>
get_base_join_indices(
    table_view const& left,
    table_view const& right,
    cudaStream_t stream) {
  CUDF_EXPECTS (0 != left.num_columns(), "Selected left dataset is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Selected right dataset is empty");
  CUDF_EXPECTS(std::equal(
      std::cbegin(left), std::cend(left),
      std::cbegin(right), std::cend(right),
      [](const auto &l, const auto &r) {
      return l.type() == r.type(); }),
      "Mismatch in joining column data types");

  constexpr join_kind BaseJoinKind = (JoinKind == join_kind::FULL_JOIN)? join_kind::LEFT_JOIN : JoinKind;
  return get_base_hash_join_indices<BaseJoinKind>(left, right, false, stream);
}

/* --------------------------------------------------------------------------*/
/**
* @brief  Combines the non common left, common left and non common right
* columns in the correct order to form the join output table.
*
* @param left_noncommon_cols Columns obtained by gathering non common left
* columns.
* @param left_noncommon_col_indices Output locations of non common left columns
* in the final table output
* @param left_common_cols Columns obtained by gathering common left
* columns.
* @param left_common_col_indices Output locations of common left columns in the
* final table output
* @param right_noncommon_cols Table obtained by gathering non common right
* columns.
*
* @Returns  Table containing rearranged columns.
*/
/* ----------------------------------------------------------------------------*/
std::vector<std::unique_ptr<column>>
combine_join_columns(
    std::vector<std::unique_ptr<column>>&& left_noncommon_cols,
    std::vector<size_type> const& left_noncommon_col_indices,
    std::vector<std::unique_ptr<column>>&& left_common_cols,
    std::vector<size_type> const& left_common_col_indices,
    std::vector<std::unique_ptr<column>>&& right_noncommon_cols) {

  std::vector<std::unique_ptr<column>> combined_cols(
      left_noncommon_cols.size() +
      left_common_cols.size());
  for(size_t i = 0; i < left_noncommon_cols.size(); ++i) {
    combined_cols.at(left_noncommon_col_indices.at(i)) =
      std::move(left_noncommon_cols.at(i));
  }
  for(size_t i = 0; i < left_common_cols.size(); ++i) {
    combined_cols.at(left_common_col_indices.at(i)) = std::move(left_common_cols.at(i));
  }
  combined_cols.insert(combined_cols.end(),
      std::make_move_iterator(right_noncommon_cols.begin()),
      std::make_move_iterator(right_noncommon_cols.end()));
  return combined_cols;
}

/* --------------------------------------------------------------------------*/
/**
* @brief  Gathers rows from `left` and `right` table and combines them into a
* single table.
*
* @param left Left input table
* @param right Right input table
* @param joined_indices Pair of vectors containing row indices from which
* `left` and `right` tables are gathered. If any row index is out of bounds,
* the contribution in the output `table` will be NULL.
* @param columns_in_common is a vector of pairs of column indices
* from tables `left` and `right` respectively, that are "in common".
* For "common" columns, only a single output column will be produced.
* For an inner or left join, the result will be gathered from the column in
* `left`. For a full join, the result will be gathered from both common
* columns in `left` and `right` and concatenated to form a single column.
*
* @Returns `table` containing the concatenation of rows from `left` and
* `right` specified by `joined_indices`.
* For any columns indicated by `columns_in_common`, only the corresponding
* column in `left` will be included in the result. Final form would look like
* `left(including common columns)+right(excluding common columns)`.
*/
/* ----------------------------------------------------------------------------*/
template <join_kind JoinKind>
std::unique_ptr<experimental::table>
construct_join_output_df(
    table_view const& left,
    table_view const& right,
    VectorPair& joined_indices,
    std::vector<std::pair<size_type, size_type>> const& columns_in_common,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
  std::vector<size_type> left_common_col;
  left_common_col.reserve(columns_in_common.size());
  std::vector<size_type> right_common_col;
  right_common_col.reserve(columns_in_common.size());
  for (const auto c : columns_in_common) {
    left_common_col.push_back(c.first);
    right_common_col.push_back(c.second);
  }
  std::vector<size_type> left_noncommon_col =
    non_common_column_indices(left.num_columns(), left_common_col);
  std::vector<size_type> right_noncommon_col =
    non_common_column_indices(right.num_columns(), right_common_col);

  bool const nullify_out_of_bounds{ JoinKind != join_kind::INNER_JOIN };

  std::unique_ptr<experimental::table> common_table;
  // Construct the joined columns
  if (not columns_in_common.empty()) {
    if (join_kind::FULL_JOIN == JoinKind) {
      auto complement_indices =
        get_left_join_indices_complement(joined_indices.second,
            left.num_rows(), right.num_rows(), stream);
      auto common_from_right = experimental::detail::gather(
          right.select(right_common_col),
          complement_indices.second.begin(),
          complement_indices.second.end(),
          false, nullify_out_of_bounds);
      auto common_from_left = experimental::detail::gather(
          left.select(left_common_col),
          joined_indices.first.begin(),
          joined_indices.first.end(),
          false, nullify_out_of_bounds);
      common_table = experimental::concatenate(
          {common_from_right->view(), common_from_left->view()});
      joined_indices =
        concatenate_vector_pairs(complement_indices, joined_indices);
    } else {
      common_table = experimental::detail::gather(
          left.select(left_common_col),
          joined_indices.first.begin(),
          joined_indices.first.end(),
          false, nullify_out_of_bounds);
    }
  }

  // Construct the left non common columns
  std::unique_ptr<experimental::table> left_table =
    experimental::detail::gather(
        left.select(left_noncommon_col),
        joined_indices.first.begin(),
        joined_indices.first.end(),
        false, nullify_out_of_bounds);

  std::unique_ptr<experimental::table> right_table =
    experimental::detail::gather(
        right.select(right_noncommon_col),
        joined_indices.second.begin(),
        joined_indices.second.end(),
        false, nullify_out_of_bounds);

  return std::make_unique<experimental::table>(
      combine_join_columns(
      left_table->release(), left_noncommon_col,
      common_table->release(), left_common_col,
      right_table->release()));
}

/* --------------------------------------------------------------------------*/
/**
 * @brief  Performs join on the columns provided in `left` and `right` as per
 * the joining indices given in `left_on` and `right_on` and creates a single
 * table.
 *
 * @throws cudf::logic_error
 * If `columns_in_common` contains a pair of indices (L, R) if L does not exist
 * in `left_on` or R does not exist in `right_on`.
 * If `columns_in_common` contains a pair of indices (L, R) such that the
 * location of `L` within `left_on` is not equal to location of R within
 * `right_on`
 * If number of elements in `left_on` or `right_on` mismatch.
 * If number of columns in either `left` or `right` table is 0 or exceeds
 * MAX_JOIN_SIZE
 * @throws std::out_of_range if element of `left_on` or `right_on` exceed the
 * number of columns in the left or right table.
 *
 * @param left The left table
 * @param right The right table
 * @param left_on The column's indices from `left` to join on.
 * Column `i` from `left_on` will be compared against column `i` of `right_on`.
 * @param right_on The column's indices from `right` to join on.
 * Column `i` from `right_on` will be compared with column `i` of `left_on`.
 * @param columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` if it is left join or from intersection of `left_on` and
 * `right_on`
 * if it is inner join or gathered from both `left_on` and `right_on` if it is
 * full join.
 * Else, for every column in `left_on` and `right_on`, an output column will
 * be produced.
 * @param mr The memory resource that will be used for allocating
 * the device memory for the new table
 * @param stream Optional, stream on which all memory allocations and copies
 * will be performed
 *
 * @tparam join_kind The type of join to be performed
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
/* ----------------------------------------------------------------------------*/
template <join_kind JoinKind>
std::unique_ptr<experimental::table>
join_call_compute_df(
    table_view const& left,
    table_view const& right,
    std::vector<size_type> const& left_on,
    std::vector<size_type> const& right_on,
    std::vector<std::pair<size_type, size_type>> const& columns_in_common,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0) {

  CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");
  CUDF_EXPECTS (left.num_rows() < MAX_JOIN_SIZE, "Left column size is too big");
  CUDF_EXPECTS (right.num_rows() < MAX_JOIN_SIZE, "Right column size is too big");

  CUDF_EXPECTS (left_on.size() == right_on.size(), "Mismatch in number of columns to be joined on");

  CUDF_EXPECTS (std::all_of(columns_in_common.begin(), columns_in_common.end(),
      [&left_on, &right_on](auto p){
      size_t lind = std::find(left_on.begin(), left_on.end(), p.first) - left_on.begin();
      size_t rind = std::find(right_on.begin(), right_on.end(), p.second) - right_on.begin();
      return (lind != left_on.size()) && (rind != right_on.size()) && (lind == rind);
      }
      ),
      "Invalid values passed to columns_in_common");

  if (is_trivial_join(left, right, left_on, right_on, JoinKind)) {
    return get_empty_joined_table(left, right, columns_in_common);
  }

  auto joined_indices =
    get_base_join_indices<JoinKind>(left.select(left_on), right.select(right_on), stream);

  return construct_join_output_df<JoinKind>(left, right, joined_indices, columns_in_common, mr, stream);
}

}

std::unique_ptr<experimental::table> inner_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                             rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::experimental::detail::join_kind::INNER_JOIN>(
        left,
        right,
        left_on,
        right_on,
        columns_in_common,
        mr);
}

std::unique_ptr<experimental::table> left_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                             rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::experimental::detail::join_kind::LEFT_JOIN>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           mr);
}

std::unique_ptr<experimental::table> full_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                         rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::experimental::detail::join_kind::FULL_JOIN>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           mr);
}

} //namespace experimental

} //namespace cudf
