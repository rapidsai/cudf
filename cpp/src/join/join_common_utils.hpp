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

#include <hash/concurrent_unordered_multimap.cuh>

#include <limits>

namespace cudf {
namespace detail {
constexpr size_type MAX_JOIN_SIZE{std::numeric_limits<size_type>::max()};

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr size_type JoinNoneValue     = std::numeric_limits<size_type>::min();

using multimap_type =
  concurrent_unordered_multimap<hash_value_type,
                                size_type,
                                size_t,
                                std::numeric_limits<hash_value_type>::max(),
                                std::numeric_limits<size_type>::max(),
                                default_hash<hash_value_type>,
                                equal_to<hash_value_type>,
                                default_allocator<thrust::pair<hash_value_type, size_type>>>;

using row_hash = cudf::row_hasher<default_hash>;

using row_equality = cudf::row_equality_comparator<true>;

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);

}  // namespace detail
}  // namespace cudf
