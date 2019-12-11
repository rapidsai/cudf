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
#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <hash/concurrent_unordered_multimap.cuh>

#include <limits>
#include <memory>
#include <algorithm>
#include <numeric>

namespace cudf {

namespace detail {

using output_index_type = ::cudf::size_type;
constexpr output_index_type MAX_JOIN_SIZE{std::numeric_limits<output_index_type>::max()};

constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;
constexpr int DEFAULT_CUDA_CACHE_SIZE = 128;
constexpr int warp_size = 32;
constexpr output_index_type JoinNoneValue = -1;

template <typename index_type>
using multimap_T =
  concurrent_unordered_multimap<hash_value_type,
                                index_type,
                                size_t,
                                std::numeric_limits<hash_value_type>::max(),
                                std::numeric_limits<index_type>::max(),
                                default_hash<hash_value_type>,
                                equal_to<hash_value_type>,
                                default_allocator< thrust::pair<hash_value_type, index_type> > >;

using RowHash =
cudf::experimental::row_hasher<default_hash>;

enum class JoinType {
  INNER_JOIN,
  LEFT_JOIN,
  FULL_JOIN,
  LEFT_SEMI_JOIN,
  LEFT_ANTI_JOIN
};

template <typename index_type>
inline
data_type
integer_type(void) {
  cudf::type_id type;
  switch(sizeof(index_type))
  {
    case 1 : type = INT8;  break;
    case 2 : type = INT16; break;
    case 4 : type = INT32; break;
    case 8 : type = INT64; break;
  }
  return data_type(type);
}

template <typename index_type>
std::unique_ptr<cudf::experimental::table>
get_empty_index_table(
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(
        std::make_unique<column>(
          integer_type<index_type>(), 0,
          rmm::device_buffer{0, stream, mr}));
    columns.emplace_back(
        std::make_unique<column>(
          integer_type<index_type>(), 0,
          rmm::device_buffer{0, stream, mr}));
    return std::make_unique<cudf::experimental::table>(std::move(columns));
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

inline auto non_common_column_indices(
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

inline std::unique_ptr<experimental::table> get_empty_joined_table(
                         table_view const& left,
                         table_view const& right,
                         std::vector<std::pair<size_type, size_type>> const& columns_in_common) {
  std::vector<size_type> right_columns_in_common (columns_in_common.size());
  for (unsigned int i = 0; i < columns_in_common.size(); ++i) {
      right_columns_in_common [i] = columns_in_common[i].second;
  }
  std::unique_ptr<experimental::table> empty_left = experimental::empty_like(left);
  std::unique_ptr<experimental::table> empty_right = experimental::empty_like(right);
  std::vector <size_type> right_non_common_indices =
    non_common_column_indices(right.num_columns(), right_columns_in_common);
  table_view tmp_right_table = (*empty_right).select(right_non_common_indices);
  table_view tmp_table{{*empty_left, tmp_right_table}};
  return std::make_unique<experimental::table>(tmp_table);
}

template <typename index_type, typename B = rmm::device_buffer>
std::unique_ptr<cudf::experimental::table>
get_indices_table(
    B&& left_indices,
    B&& right_indices,
    cudf::size_type join_size,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  rmm::device_buffer l{std::forward<B>(left_indices)};
  rmm::device_buffer r{std::forward<B>(right_indices)};
  l.resize(sizeof(index_type)*join_size, stream);
  r.resize(sizeof(index_type)*join_size, stream);
  //l.shrink_to_fit(stream);//TODO : probably not needed since indices are temp for the duration of join
  //r.shrink_to_fit(stream);//TODO : probably not needed since indices are temp for the duration of join
  std::vector<std::unique_ptr<column>> columns;
  columns.emplace_back(
      std::make_unique<column>(
        integer_type<index_type>(), join_size,
        std::move(l)));
  columns.emplace_back(
      std::make_unique<column>(
        integer_type<index_type>(), join_size,
        std::move(r)));
  return std::make_unique<cudf::experimental::table>(std::move(columns));
}

inline bool is_trivial_join(table_view const& left,
                            table_view const& right,
                            std::vector<size_type> const& left_on,
                            std::vector<size_type> const& right_on,
                            JoinType join_type) {

  // If there is nothing to join, then send empty table with all columns
  if (left_on.empty() || right_on.empty()) {
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

  // If left semi join (contains) and right table is empty,
  // return immediately
  if ((JoinType::LEFT_SEMI_JOIN == join_type) &&
      (0 == right.num_rows())) {
    return true;
  }

  return false;
}


}//namespace detail

}//namespace cudf
