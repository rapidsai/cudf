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

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {
/// Definition of the opaque base class
class group_labels {
 protected:
  enum Method { HASH, SORT };
  group_labels(group_labels::Method _method) : method{_method} {}
  Method method;  ///< Indicates the method used to compute the aggregation
};

/// Derived type for group labels from a hash-based groupby
class hash_group_labels : public group_labels {
 public:
  hash_group_labels() : group_labels{group_labels::HASH} {}
  /// For a hash-based groupby, the group_labels contains the actual table of
  /// unique keys
  std::unique_ptr<experimental::table> unique_keys{};
};

/// Derived type for group labels from a sort-based groupby
class sort_group_labels : public group_labels {
 public:
  sort_group_labels() : group_labels{group_labels::SORT} {}
  /// For sort-based groupby, we can do something smarter.
};

namespace detail {
namespace hash {
/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param keys The table of keys
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby can be used
 * @return false A hash-based groupby cannot be used
 */
bool use_hash_groupby(table_view const& keys,
                      std::vector<aggregation_request> const& requests);

// Hash-based groupby
std::pair<std::unique_ptr<group_labels>, std::vector<aggregation_result>>
groupby(table_view const& keys,
        std::vector<aggregation_request> const& requests, bool ignore_null_keys,
        cudaStream_t stream, rmm::mr::device_memory_resource* mr);
}  // namespace hash

namespace sort {
// Sort-based groupby
std::pair<std::unique_ptr<group_labels>, std::vector<aggregation_result>>
groupby(table_view const& keys,
        std::vector<aggregation_request> const& requests, cudaStream_t stream,
        rmm::mr::device_memory_resource* mr);
}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
