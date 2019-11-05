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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {
namespace hash {
namespace detail {

/**
 * @brief List of aggregation operations that can be computed with a hash-based
 * implementation.
 */
static constexpr std::array<aggregation_request::Kind, 5> hash_aggregations{
    aggregation_request::SUM, aggregation_request::MIN,
    aggregation_request::MAX, aggregation_request::COUNT,
    aggregation_request::MEAN};

template <class T, size_t N>
constexpr bool array_contains(std::array<T, N> const& haystack, T needle) {
  for (auto i = 0u; i < N; ++i) {
    if (haystack[i] == needle) return true;
  }
  return false;
}

constexpr bool is_hash_aggregation(aggregation_request::Kind t) {
  return array_contains(hash_aggregations, t);
}

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby(
    table_view const& keys,
    std::vector<std::unique_ptr<aggregation_request>> const& requests,
    Options options, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  CUDF_EXPECTS(std::all_of(requests.begin(), requests.end(),
                           [](auto const& r) {
                             return is_hash_aggregation(r->aggregation);
                           }),
               "Invalid aggregation for hash-based groupby.");

  return std::make_pair(std::make_unique<table>(), std::make_unique<table>());
}
}  // namespace detail

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> groupby(
    table_view const& keys,
    std::vector<std::unique_ptr<aggregation_request>> const& requests,
    Options options, rmm::mr::device_memory_resource* mr) {
  return detail::groupby(keys, requests, options, 0, mr);
}
}  // namespace hash
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf