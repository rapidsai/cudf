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

#include <cudf/copying.hpp>
#include <cudf/join_new.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/gather.cuh>

#include <join/join_common_utils.hpp>
#include <join/nested_loop_join.cuh>

namespace cudf {

namespace join {

namespace detail {

  // TODO:
  //  nested_join_indices      - my function/kernel
  //  sort_merge_join_indices  - Kumar
  //  hash_join_indices        - rework current hash mechanism
  //  filter_join_indices      - new function/kernel
  //
  struct nested_loop_join {
    rmm::device_vector<int64_t> operator()(table_view const& left,
                                           table_view const& right,
                                           std::vector<join_operation> const& primary_join_ops,
                                           std::vector<join_operation> const& secondary_join_ops,
                                           cudaStream_t stream) {

      return nested_join_indices(left, right, primary_join_ops, stream);
    }
  };

  struct sort_merge_join {
    rmm::device_vector<int64_t> operator()(table_view const& left,
                                           table_view const& right,
                                           std::vector<join_operation> const& primary_join_ops,
                                           std::vector<join_operation> const& secondary_join_ops,
                                           cudaStream_t stream) {

      /*
      auto joined_indices = sort_merge_join_indices(left, right, primary_join_ops, stream);

      if (secondary_join_ops.empty()) {
        return joined_indices;
      }

      return filter_join_indices(left, right, joined_indices, secondary_join_ops, stream);
      */
      return rmm::device_vector<int64_t>{};
    }
  };

  struct hash_join {
    rmm::device_vector<int64_t> operator()(table_view const& left,
                                           table_view const& right,
                                           std::vector<join_operation> const& primary_join_ops,
                                           std::vector<join_operation> const& secondary_join_ops,
                                           cudaStream_t stream) {

      /*
      auto joined_indices = hash_join_indices(left, right, primary_join_ops, stream);

      if (secondary_join_ops.empty()) {
        return joined_indices;
      }

      return filter_join_indices(left, right, joined_indices, secondary_join_ops, stream);
      */
      return rmm::device_vector<int64_t>{};
    }
  };
    
  bool is_trivial_join(table_view const& left,
                       table_view const& right,
                       std::vector<join_operation> const& primary_join_ops,
                       std::vector<join_operation> const& secondary_join_ops,
                       join_kind JoinKind) {

    // If there is nothing to join, then send empty table with all columns
    if (primary_join_ops.empty() && secondary_join_ops.empty()) {
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
  auto non_common_column_indices(size_type num_columns,
                                 std::vector<size_type> const& common_column_indices) {
    CUDF_EXPECTS(common_column_indices.size() <= static_cast<unsigned long>(num_columns),
                 "Too many columns in common");
    std::vector<size_type> all_column_indices(num_columns);
    std::iota(std::begin(all_column_indices), std::end(all_column_indices), 0);
    std::vector<size_type> sorted_common_column_indices{common_column_indices};
    std::sort(std::begin(sorted_common_column_indices), std::end(sorted_common_column_indices));
    std::vector<size_type> non_common_column_indices(num_columns - common_column_indices.size());
    std::set_difference(std::cbegin(all_column_indices),
                        std::cend(all_column_indices),
                        std::cbegin(sorted_common_column_indices),
                        std::cend(sorted_common_column_indices), std::begin(non_common_column_indices));
    return non_common_column_indices;
  }

  std::unique_ptr<experimental::table> get_empty_joined_table(table_view const& left,
                                                              table_view const& right,
                                                              std::vector<std::pair<size_type, size_type>> const& columns_in_common) {
    std::vector<size_type> right_columns_in_common (columns_in_common.size());
    std::transform(columns_in_common.begin(),
                   columns_in_common.end(),
                   right_columns_in_common.begin(),
                   [](auto& col) { return col.second; } );
    std::unique_ptr<experimental::table> empty_left = experimental::empty_like(left);
    std::unique_ptr<experimental::table> empty_right = experimental::empty_like(right);
    std::vector <size_type> right_non_common_indices = non_common_column_indices(right.num_columns(), right_columns_in_common);
    table_view tmp_right_table = (*empty_right).select(right_non_common_indices);
    table_view tmp_table{{*empty_left, tmp_right_table}};
    return std::make_unique<experimental::table>(tmp_table);
  }
  
  template <typename iterator>
  rmm::device_vector<cudf::size_type> get_indices_complement(iterator indices_begin,
                                                             cudf::size_type join_size,
                                                             cudf::size_type row_count,
                                                             cudaStream_t stream) {

    //Get array of indices that do not appear in indices

    //Vector allocated for unmatched result
    rmm::device_vector<cudf::size_type> indices_complement(row_count);

    //
    // NOTE:  In this implementation, the indices_begin and indices_end
    //        iterators point to a range of elements from an INNER_JOIN,
    //        meaning they only include valid values within the table
    //
    rmm::device_vector<cudf::size_type> invalid_index_map(row_count, 1);

    //invalid_index_map[index_ptr[i]] = 0 for i = 0 to row_count
    //Thus specifying that those locations are valid
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    thrust::make_constant_iterator(0),
                    thrust::make_constant_iterator(0) + join_size,
                    indices_begin,                      //Index locations
                    invalid_index_map.begin());         //Output indices

    //Create list of indices that have been marked as invalid
    auto copy_end = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                                    thrust::make_counting_iterator<cudf::size_type>(0),
                                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                                    invalid_index_map.begin(),
                                    indices_complement.begin(),
                                    thrust::identity<size_type>());

    cudf::size_type indices_count = thrust::distance(indices_complement.begin(), copy_end);

    indices_complement.resize(indices_count);
    return indices_complement;
  }

  std::vector<std::unique_ptr<column>> combine_join_columns(std::vector<std::unique_ptr<column>>&& left_noncommon_cols,
                                                            std::vector<size_type> const& left_noncommon_col_indices,
                                                            std::vector<std::unique_ptr<column>>&& left_common_cols,
                                                            std::vector<size_type> const& left_common_col_indices,
                                                            std::vector<std::unique_ptr<column>>&& right_noncommon_cols) {

    std::vector<std::unique_ptr<column>> combined_cols(left_noncommon_cols.size() + left_common_cols.size());
    for(size_t i = 0; i < left_noncommon_cols.size(); ++i) {
      combined_cols.at(left_noncommon_col_indices.at(i)) = std::move(left_noncommon_cols.at(i));
    }
    for(size_t i = 0; i < left_common_cols.size(); ++i) {
      combined_cols.at(left_common_col_indices.at(i)) = std::move(left_common_cols.at(i));
    }
    combined_cols.insert(combined_cols.end(),
                         std::make_move_iterator(right_noncommon_cols.begin()),
                         std::make_move_iterator(right_noncommon_cols.end()));
    return combined_cols;
  }

  // TODO:  Idea is to use this function and make_transform_iterator to
  //        do the proper gather.
    //  THINKING ABOUT THIS...
    //     seems like my best option is to create a custom iterator
    //     (the current CUSTOM_ITERATOR) would just divide and modulo
    //     by the number of rows in right table.
    //
    //     I would need another custom iterator that iterates over
    //     3 data structures:  joined_indices, left_indices_complement
    //     and right_indices_complement.  While we're in inner_join
    //     it does what the original custom iterator did.  Once
    //     we reach the end of that we iterate over left.  If we're
    //     computing right then we return JoinNoneValue.  If we're
    //     computing left then we return the value.  Finally we iterate
    //     over left doing the reverse.
    //
    //     This would allow us to only use O(n+m) extra memory for the
    //     indices complement arrays and use this new custom iterator
    //     in the gather phase at the end.
    //
  template <bool left_index>
  struct join_output_iterator {
    __host__ join_output_iterator(rmm::device_vector<int64_t> &joined_indices,
                                  rmm::device_vector<cudf::size_type> &left_complement,
                                  rmm::device_vector<cudf::size_type> &right_complement,
                                  cudf::size_type right_num_rows):
      _joined_indices(joined_indices.data().get()),
      _left_complement(left_complement.data().get()),
      _right_complement(right_complement.data().get()),
      _joined_indices_size(joined_indices.size()),
      _left_complement_size(left_complement.size()),
      _right_complement_size(right_complement.size()),
      _right_num_rows(right_num_rows)  {}
    
    __device__ cudf::size_type operator()(cudf::size_type index) {
      if (index < _joined_indices_size) {
        if (left_index) {
          return _joined_indices[index] / _right_num_rows;
        } else {
          return _joined_indices[index] % _right_num_rows;
        }
      }

      index -= _joined_indices_size;
      if (index < _left_complement_size) {
        if (left_index) {
          return _left_complement[index];
        } else {
          return JoinNoneValue;
        }
      }

      index -= _left_complement_size;
      if (left_index) {
        return JoinNoneValue;
      } else {
        return _right_complement[index];
      }
    }

  private:
    int64_t          *_joined_indices;
    cudf::size_type  *_left_complement;
    cudf::size_type  *_right_complement;
    cudf::size_type   _joined_indices_size;
    cudf::size_type   _left_complement_size;
    cudf::size_type   _right_complement_size;
    cudf::size_type   _right_num_rows;
  };
  
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
  template <join_kind JoinKind>
  std::unique_ptr<experimental::table> construct_join_output_df(table_view const& left,
                                                                table_view const& right,
                                                                rmm::device_vector<int64_t> & joined_indices,
                                                                std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                                                                rmm::mr::device_memory_resource* mr,
                                                                cudaStream_t stream) {

    if ((join_kind::INNER_JOIN == JoinKind) && (joined_indices.size() == 0)) {
      return get_empty_joined_table(left, right, columns_in_common);
    }
    
    std::vector<size_type> left_common_col;
    left_common_col.reserve(columns_in_common.size());
    std::vector<size_type> right_common_col;
    right_common_col.reserve(columns_in_common.size());
    for (const auto c : columns_in_common) {
      left_common_col.push_back(c.first);
      right_common_col.push_back(c.second);
    }
    std::vector<size_type> left_noncommon_col = non_common_column_indices(left.num_columns(), left_common_col);
    std::vector<size_type> right_noncommon_col = non_common_column_indices(right.num_columns(), right_common_col);

    //
    //   inner_join_indices only gives us part of the solution if we want
    //   left join or right join.  So we need to get what we need for the output
    //   dataframe.
    //
    rmm::device_vector<cudf::size_type> left_indices_complement{};
    rmm::device_vector<cudf::size_type> right_indices_complement{};

    join_output_iterator<true>  left_iterator(joined_indices, left_indices_complement, right_indices_complement, right.num_rows());
    join_output_iterator<false> right_iterator(joined_indices, left_indices_complement, right_indices_complement, right.num_rows());
    
    cudf::size_type output_size = joined_indices.size();

    bool const nullify_out_of_bounds{ JoinKind != join_kind::INNER_JOIN };

    if (join_kind::LEFT_JOIN == JoinKind) {
      left_indices_complement = get_indices_complement(thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), left_iterator),
                                                       output_size,
                                                       left.num_rows(),
                                                       stream);
      output_size += left_indices_complement.size();
    }

    if (join_kind::FULL_JOIN == JoinKind) {
      left_indices_complement = get_indices_complement(thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), left_iterator),
                                                       output_size,
                                                       left.num_rows(),
                                                       stream);
      right_indices_complement = get_indices_complement(thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), right_iterator),
                                                        output_size,
                                                        right.num_rows(),
                                                        stream);

      output_size += left_indices_complement.size() + right_indices_complement.size();
    }

    std::unique_ptr<experimental::table> common_table;

    //
    //  Update in case they changed above
    //
    left_iterator = join_output_iterator<true>(joined_indices, left_indices_complement, right_indices_complement, right.num_rows());
    right_iterator = join_output_iterator<false>(joined_indices, left_indices_complement, right_indices_complement, right.num_rows());

    // Construct the joined columns
    if (not columns_in_common.empty()) {
      if (join_kind::FULL_JOIN == JoinKind) {
        auto common_from_right = experimental::detail::gather(right.select(right_common_col),
                                                              right_indices_complement.begin(),
                                                              right_indices_complement.end(),
                                                              false, nullify_out_of_bounds);
        auto common_from_left = experimental::detail::gather(left.select(left_common_col),
                                                             thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), left_iterator),
                                                             thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(joined_indices.size() + left_indices_complement.size()), left_iterator),
                                                             false, nullify_out_of_bounds);

        common_table = experimental::concatenate({common_from_left->view(), common_from_right->view()});
      } else {
        common_table = experimental::detail::gather(left.select(left_common_col),
                                                    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), left_iterator),
                                                    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(output_size), left_iterator),
                                                    false, nullify_out_of_bounds);
      }
    }

    // Construct the left non common columns
    auto left_table = experimental::detail::gather(left.select(left_noncommon_col),
                                                   thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), left_iterator),
                                                   thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(output_size), left_iterator),
                                                   false, nullify_out_of_bounds);

    auto right_table = experimental::detail::gather(right.select(right_noncommon_col),
                                                    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0), right_iterator),
                                                    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(output_size), right_iterator),
                                                    false, nullify_out_of_bounds);

    return std::make_unique<experimental::table>(combine_join_columns(left_table->release(), left_noncommon_col,
                                                                      common_table->release(), left_common_col,
                                                                      right_table->release()));
  }
  
  template <join_kind JoinKind, typename join_indices_type>
  std::unique_ptr<experimental::table> join(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<join_operation> const& primary_join_ops,
                                            std::vector<join_operation> const& secondary_join_ops,
                                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                            join_indices_type join_indices_impl,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream = 0) {

    CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
    CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");

    for (auto p : columns_in_common) {
      CUDF_EXPECTS(left.column(p.first).type() == right.column(p.second).type(), "Mismatch in columns in common data types");
    }

    for (auto p : primary_join_ops) {
      CUDF_EXPECTS(left.column(p.left_column_idx).type() == right.column(p.right_column_idx).type(), "Mismatch in primary_join_ops data types");
    }

    for (auto p : secondary_join_ops) {
      CUDF_EXPECTS(left.column(p.left_column_idx).type() == right.column(p.right_column_idx).type(), "Mismatch in secondary_join_ops data types");
    }

    if (is_trivial_join(left, right, primary_join_ops, secondary_join_ops, JoinKind)) {
      return get_empty_joined_table(left, right, columns_in_common);
    }

    if ((join_kind::FULL_JOIN == JoinKind) && (0 == left.num_rows())) {
      //
      //  write this example.  Return a table with nulls for left columns and all of the
      //  right columns
      //
    }

    //
    //  Call the specific join implementation.  Note that in the new implementation join_indices_impl
    //  only computes the INNER_JOIN.  If a LEFT_JOIN or FULL_JOIN is desired that is addressed
    //  in construction the output data frame below.
    //
    auto joined_indices = join_indices_impl(left, right, primary_join_ops, secondary_join_ops, stream);

    return construct_join_output_df<JoinKind>(left, right, joined_indices, columns_in_common, mr, stream);
  }
} //namespace detail

namespace inner_join {
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr) {

    return detail::join<detail::join_kind::INNER_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::INNER_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<join_operation> const& primary_join_ops,
                                            std::vector<join_operation> const& secondary_join_ops,
                                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                            rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::INNER_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }
} //namespace inner_join

namespace left_join {
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<join_operation> const& primary_join_ops,
                                            std::vector<join_operation> const& secondary_join_ops,
                                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                            rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }

} //namespace left_join

namespace full_join {
  std::unique_ptr<experimental::table> nested_loop(cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   std::vector<join_operation> const& join_ops,
                                                   std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                   rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<experimental::table> sort_merge(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<experimental::table> hash(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<join_operation> const& primary_join_ops,
                                            std::vector<join_operation> const& secondary_join_ops,
                                            std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                            rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }
} //namespace full_join

namespace nested_loop {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::INNER_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, join_ops, join_ops, columns_in_common, detail::nested_loop_join{}, mr);
  }
} //namespace nested_loop

namespace sort_merge {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::INNER_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::sort_merge_join{}, mr);
  }
} //namespace sort_merge
  
namespace hash {
  std::unique_ptr<experimental::table> inner_join(cudf::table_view const& left,
                                                  cudf::table_view const& right,
                                                  std::vector<join_operation> const& primary_join_ops,
                                                  std::vector<join_operation> const& secondary_join_ops,
                                                  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                  rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::INNER_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }

  std::unique_ptr<experimental::table> left_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::LEFT_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }

  std::unique_ptr<experimental::table> full_join(cudf::table_view const& left,
                                                 cudf::table_view const& right,
                                                 std::vector<join_operation> const& primary_join_ops,
                                                 std::vector<join_operation> const& secondary_join_ops,
                                                 std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                                                 rmm::mr::device_memory_resource* mr) {
    return detail::join<detail::join_kind::FULL_JOIN>(left, right, primary_join_ops, secondary_join_ops, columns_in_common, detail::hash_join{}, mr);
  }
} //namespace hash
  
} //namespace join

} //namespace cudf
