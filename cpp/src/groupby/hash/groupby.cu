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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {
namespace hash {
namespace {
// This is a temporary fix due to compiler bug and we can resort back to
// constexpr once cuda 10.2 becomes RAPIDS's minimum compiler version
#if 0
/**
 * @brief List of aggregation operations that can be computed with a hash-based
 * implementation.
 */
constexpr std::array<aggregation::Kind, 5> hash_aggregations{
    aggregation::SUM, aggregation::MIN, aggregation::MAX, aggregation::COUNT,
    aggregation::MEAN};

template <class T, size_t N>
constexpr bool array_contains(std::array<T, N> const& haystack, T needle) {
  for (auto i = 0u; i < N; ++i) {
    if (haystack[i] == needle) return true;
  }
  return false;
}
#endif

/**
 * @brief Indicates whether the specified aggregation operation can be computed
 * with a hash-based implementation.
 *
 * @param t The aggregation operation to verify
 * @return true `t` is valid for a hash based groupby
 * @return false `t` is invalid for a hash based groupby
 */
bool constexpr is_hash_aggregation(aggregation::Kind t) {
  // this is a temporary fix due to compiler bug and we can resort back to
  // constexpr once cuda 10.2 becomes RAPIDS's minimum compiler version
  // return array_contains(hash_aggregations, t);
  return (t == aggregation::SUM) or (t == aggregation::MIN) or
         (t == aggregation::MAX) or (t == aggregation::COUNT) or
         (t == aggregation::MEAN);
}
}  // namespace

/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param keys The table of keys
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby should be used
 * @return false A hash-based groupby should not be used
 */
bool can_use_hash_groupby(table_view const& keys,
                      std::vector<aggregation_request> const& requests) {
  return false;
  // TODO (dm): Jake to enable the following code when hash-groupby is ready
  // return std::all_of(
  //     requests.begin(), requests.end(), [](aggregation_request const& r) {
  //       return std::all_of(
  //           r.aggregations.begin(), r.aggregations.end(),
  //           [](auto const& a) { return is_hash_aggregation(a->kind); });
  //     });
}

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    bool ignore_null_keys, cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {
  // stub
  return std::make_pair(std::make_unique<table>(),
                        std::vector<aggregation_result>{});
}
}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
