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

#include "hash_join.cuh"

namespace cudf {

namespace detail {

bool is_trivial_join(
                     table_view const& left,
                     table_view const& right,
                     std::vector<size_type> const& left_on,
                     std::vector<size_type> const& right_on,
                     JoinType join_type) {
  // If there is nothing to join, then send empty table with all columns
  if (left_on.empty() || right_on.empty() || left_on.size() != right_on.size()) {
      return true;
  }

  // Even though the resulting table might be empty, but the column should match the expected dtypes and other necessary information
  // So, there is a possibility that there will be lesser number of right columns, so the tmp_table.
  // If the inputs are empty, immediately return
  if ((0 == left.num_rows()) && (0 == right.num_rows())) {
      return true;
  }

  // If left join and the left table is empty, return immediately
  if ((JoinType::LEFT_JOIN == join_type) && (0 == left.num_rows())) {
      return true;
  }

  // If Inner Join and either table is empty, return immediately
  if ((JoinType::INNER_JOIN == join_type) &&
      ((0 == left.num_rows()) || (0 == right.num_rows()))) {
      return true;
  }

  return false;
}

template <typename index_type>
std::unique_ptr<experimental::table>
get_trivial_full_join_indices(
    table_view const& left,
    table_view const& right,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS((left.num_rows() != 0) || (right.num_rows() != 0),
      "Expecting at least one table view to have non-zero number of rows");

  rmm::device_buffer left_index, right_index;
  size_type result_size = 0;

  if (left.num_rows() == 0) {
    result_size = right.num_rows();
    left_index  = rmm::device_buffer{sizeof(index_type)*result_size, stream, mr};
    thrust::device_ptr<index_type> l(static_cast<index_type*>(left_index.data()));
    thrust::fill(rmm::exec_policy(stream)->on(stream), l, l + result_size,
        static_cast<index_type>(-1));

    right_index = rmm::device_buffer{sizeof(index_type)*result_size, stream, mr};
    thrust::device_ptr<index_type> r(static_cast<index_type*>(right_index.data()));
    thrust::sequence(rmm::exec_policy(stream)->on(stream), r, r + result_size);
  } else if (right.num_rows() == 0) {
    result_size = left.num_rows();
    left_index  = rmm::device_buffer{sizeof(index_type)*result_size, stream, mr};
    thrust::device_ptr<index_type> l(static_cast<index_type*>(left_index.data()));
    thrust::sequence(rmm::exec_policy(stream)->on(stream), l, l + result_size);

    right_index = rmm::device_buffer{sizeof(index_type)*result_size, stream, mr};
    thrust::device_ptr<index_type> r(static_cast<index_type*>(right_index.data()));
    thrust::fill(rmm::exec_policy(stream)->on(stream), r, r + result_size,
        static_cast<index_type>(-1));
  }

  return get_indices_table<index_type>(
      std::move(left_index), std::move(right_index),
      result_size, stream, mr);
}

template <JoinType join_type, typename index_type>
std::unique_ptr<experimental::table>
get_base_join_indices(
    table_view const& left,
    table_view const& right,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS (0 != left.num_columns(), "Selected left dataset is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Selected right dataset is empty");
  CUDF_EXPECTS(std::equal(
      std::cbegin(left), std::cend(left),
      std::cbegin(right), std::cend(right),
      [](const auto &l, const auto &r) {
      return l.type() == r.type(); }),
      "Mismatch in joining column data types");

  //TODO : This can probably be removed since get_trivial_full_join_indices is split between
  //get_base_hash_join_indices and get_join_indices_complement
  bool is_trivial_full_join =
    (JoinType::FULL_JOIN == join_type) &&
    ((0 == left.num_rows()) || (0 == right.num_rows()));
  if (is_trivial_full_join) { return get_trivial_full_join_indices<index_type>(left, right, stream, mr); }

  constexpr JoinType base_join_type = (join_type == JoinType::FULL_JOIN)? JoinType::LEFT_JOIN : join_type;
  return get_base_hash_join_indices<base_join_type, index_type>(left, right, false, stream, mr);
}

std::vector<std::unique_ptr<column>>
combine_join_columns(
    std::unique_ptr<experimental::table> left_table,
    std::vector<size_type>& left_noncommon_col,
    std::unique_ptr<experimental::table> common_table,
    std::vector<size_type>& left_common_col,
    std::unique_ptr<experimental::table> right_table,
    std::vector<size_type>& right_noncommon_col) {
  std::vector<std::unique_ptr<column>> left_cols;
  std::vector<std::unique_ptr<column>> common_cols;
  std::vector<std::unique_ptr<column>> right_cols;
  if (left_table != nullptr) {
    left_cols = std::move(left_table->release());
  }
  if (common_table != nullptr) {
    common_cols = std::move(common_table->release());
  }
  if (right_table != nullptr) {
    right_cols = std::move(right_table->release());
  }
  std::vector<std::unique_ptr<column>> combined_cols(
      left_cols.size() + right_cols.size() + common_cols.size());
  for(size_t i = 0; i < left_cols.size(); ++i) {
    combined_cols.at(left_noncommon_col.at(i)) = std::move(left_cols.at(i));
  }
  for(size_t i = 0; i < common_cols.size(); ++i) {
    combined_cols.at(left_common_col.at(i)) = std::move(common_cols.at(i));
  }
  for(size_t i = 0; i < right_cols.size(); ++i) {
    combined_cols.at(right_noncommon_col.at(i)) = std::move(right_cols.at(i));
  }
  return std::move(combined_cols);
}

template <JoinType join_type, typename index_type>
std::unique_ptr<experimental::table>
construct_join_output_df(
    table_view const& left,
    table_view const& right,
    std::unique_ptr<experimental::table> joined_indices,
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

  bool const ignore_out_of_bounds{ join_type != JoinType::INNER_JOIN };

  std::unique_ptr<experimental::table> common_table;
  // Construct the joined columns
  if (not columns_in_common.empty()) {
    if (JoinType::FULL_JOIN == join_type) {
      auto complement_indices =
        get_join_indices_complement<index_type>(joined_indices->get_column(1),
            left.num_rows(), right.num_rows(), mr, stream);
      auto common_from_right = experimental::detail::gather(
          right.select(right_common_col),
          complement_indices->get_column(1),
          false, ignore_out_of_bounds);
      auto common_from_left = experimental::detail::gather(
          left.select(left_common_col),
          joined_indices->get_column(0),
          false, ignore_out_of_bounds);
      common_table = experimental::concatenate(
          {common_from_right->view(), common_from_left->view()});
      joined_indices = experimental::concatenate(
          {complement_indices->view(), joined_indices->view()});
    } else {
      common_table = experimental::detail::gather(
          left.select(left_common_col),
          joined_indices->get_column(0),
          false, ignore_out_of_bounds);
    }
  }

  std::unique_ptr<experimental::table> left_table;
  // Construct the left non common columns
  if (not left_noncommon_col.empty()) {
    left_table = experimental::detail::gather(
        left.select(left_noncommon_col),
        joined_indices->get_column(0),
        false, ignore_out_of_bounds);
  }

  std::unique_ptr<experimental::table> right_table;
  // Construct the right non common columns
  if (not right_noncommon_col.empty()) {
    right_table = experimental::detail::gather(
        right.select(right_noncommon_col),
        joined_indices->get_column(1),
        false, ignore_out_of_bounds);
  }
  return std::make_unique<experimental::table>(
      combine_join_columns(
      std::move(left_table), left_noncommon_col,
      std::move(common_table), left_common_col,
      std::move(right_table), right_noncommon_col));
}

template <JoinType join_type, typename index_type>
std::unique_ptr<experimental::table>
join_call_compute_df(
    table_view const& left,
    table_view const& right,
    std::vector<size_type> const& left_on,
    std::vector<size_type> const& right_on,
    std::vector<std::pair<size_type, size_type>> const& columns_in_common,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");
  CUDF_EXPECTS( left.num_rows() < MAX_JOIN_SIZE, "Left column size is too big");
  CUDF_EXPECTS( right.num_rows() < MAX_JOIN_SIZE, "Right column size is too big");

  CUDF_EXPECTS (left_on.size() == right_on.size(), "Mismatch in number of columns to be joined on");
  //TODO : return empty table if left_on or right_on empty or their sizes mismatch

  if (is_trivial_join(left, right, left_on, right_on, join_type)) {
    return get_empty_joined_table(left, right, columns_in_common);
  }

  std::unique_ptr<cudf::experimental::table> joined_indices =
    get_base_join_indices<join_type, index_type>(left.select(left_on), right.select(right_on), stream, mr);

  return construct_join_output_df<join_type, index_type>(left, right, std::move(joined_indices), columns_in_common, mr, stream);
}

}

std::unique_ptr<experimental::table> inner_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                             cudaStream_t stream,
                             rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::detail::JoinType::INNER_JOIN, ::cudf::detail::output_index_type>(
        left,
        right,
        left_on,
        right_on,
        columns_in_common,
        stream,
        mr);
}

std::unique_ptr<experimental::table> left_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                             cudaStream_t stream,
                             rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::detail::JoinType::LEFT_JOIN, ::cudf::detail::output_index_type>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           stream,
           mr);
}

std::unique_ptr<experimental::table> full_join(
                             table_view const& left,
                             table_view const& right,
                             std::vector<size_type> const& left_on,
                             std::vector<size_type> const& right_on,
                             std::vector<std::pair<size_type, size_type>> const& columns_in_common,
                             cudaStream_t stream,
                         rmm::mr::device_memory_resource* mr) {
    return detail::join_call_compute_df<::cudf::detail::JoinType::FULL_JOIN, ::cudf::detail::output_index_type>(
           left,
           right,
           left_on,
           right_on,
           columns_in_common,
           stream,
           mr);
}

}
