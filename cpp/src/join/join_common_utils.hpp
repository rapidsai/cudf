/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

#include <cuco/static_multimap.cuh>
#include <hash/concurrent_unordered_multimap.cuh>

#include <limits>

namespace cudf {
namespace detail {
constexpr size_type MAX_JOIN_SIZE{std::numeric_limits<size_type>::max()};

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr size_type JoinNoneValue     = std::numeric_limits<size_type>::min();

using pair_type = cuco::pair_type<hash_value_type, size_type>;

using multimap_type = cuco::static_multimap<hash_value_type,
                                            size_type,
                                            cuco::double_hashing<hash_value_type, size_type>,
                                            cuda::thread_scope_device,
                                            default_allocator<pair_type>>;

using row_hash = cudf::row_hasher<default_hash>;

using row_equality = cudf::row_equality_comparator<true>;

class pair_equality {
 public:
  pair_equality(table_device_view lhs, table_device_view rhs, bool nulls_are_equal = true)
    : _check_row_equality{lhs, rhs, nulls_are_equal}
  {
  }

  __device__ __inline__ bool operator()(const pair_type& lhs, const pair_type& rhs) const noexcept
  {
    bool res = (lhs.first == rhs.first);
    if (res) { return _check_row_equality(rhs.second, lhs.second); }
    return res;
  }

 private:
  cudf::row_equality_comparator<true> _check_row_equality;
};

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

inline bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  // If there is nothing to join, then send empty table with all columns
  if (left.is_empty() || right.is_empty()) { return true; }

  // If left join and the left table is empty, return immediately
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }

  // If Inner Join and either table is empty, return immediately
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }

  // If left semi join (contains) and right table is empty,
  // return immediately
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }

  return false;
}

}  // namespace detail

}  // namespace cudf
