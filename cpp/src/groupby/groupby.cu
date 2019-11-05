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
#include <cudf/detail/groupby.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {
// Dispatch to hash vs. sort groupby
std::pair<std::unique_ptr<table>, std::vector<std::unique_ptr<column>>> groupby(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    Options options, cudaStream_t stream, rmm::mr::device_memory_resource* mr) {
  if (hash::use_hash_groupby(keys, requests, options)) {
    return hash::groupby(keys, requests, options, stream, mr);
  } else {
    return sort::groupby(keys, requests, options, stream, mr);
  }
}
}  // namespace detail

// Public groupby interface
std::pair<std::unique_ptr<table>, std::vector<std::unique_ptr<column>>> groupby(
    table_view const& keys, std::vector<aggregation_request> const& requests,
    Options options, rmm::mr::device_memory_resource* mr) {
  return detail::groupby(keys, requests, options, 0, mr);
}
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf