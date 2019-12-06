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

#include <limits>
#include <memory>
#include <algorithm>
#include <numeric>

namespace cudf {

namespace experimental {

namespace detail {

using output_index_type = ::cudf::size_type;
constexpr output_index_type MAX_JOIN_SIZE{std::numeric_limits<output_index_type>::max()};

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr output_index_type JoinNoneValue = -1;

template <typename index_type>
using multimap_t =
  concurrent_unordered_multimap<hash_value_type,
                                index_type,
                                size_t,
                                std::numeric_limits<hash_value_type>::max(),
                                std::numeric_limits<index_type>::max(),
                                default_hash<hash_value_type>,
                                equal_to<hash_value_type>,
                                default_allocator< thrust::pair<hash_value_type, index_type> > >;

using row_hash =
cudf::experimental::row_hasher<default_hash>;

enum class join_kind {
  INNER_JOIN,
  LEFT_JOIN,
  FULL_JOIN
};


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

template <typename index_type>
std::pair<rmm::device_vector<index_type>,
rmm::device_vector<index_type>>
concatenate_vector_pairs(
std::pair<rmm::device_vector<index_type>,
rmm::device_vector<index_type>>& a,
std::pair<rmm::device_vector<index_type>,
rmm::device_vector<index_type>>& b)
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


}//namespace detail

} //namespace experimental

}//namespace cudf
