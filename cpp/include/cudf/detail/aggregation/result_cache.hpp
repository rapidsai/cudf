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

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <unordered_map>

namespace cudf {
namespace detail {
struct aggregation_equality {
  bool operator()(aggregation const& lhs, aggregation const& rhs) const
  {
    return lhs.is_equal(rhs);
  }
};

struct aggregation_hash {
  size_t operator()(aggregation const& key) const noexcept { return key.do_hash(); }
};

class result_cache {
 public:
  result_cache()                    = delete;
  ~result_cache()                   = default;
  result_cache(result_cache const&) = delete;
  result_cache& operator=(const result_cache& other) = delete;

  result_cache(size_t num_columns) : _cache(num_columns) {}

  bool has_result(size_t col_idx, aggregation const& agg) const;

  void add_result(size_t col_idx, aggregation const& agg, std::unique_ptr<column>&& col);

  column_view get_result(size_t col_idx, aggregation const& agg) const;

  std::unique_ptr<column> release_result(size_t col_idx, aggregation const& agg);

 private:
  std::vector<std::unordered_map<std::reference_wrapper<aggregation const>,
                                 std::pair<std::unique_ptr<aggregation>, std::unique_ptr<column>>,
                                 aggregation_hash,
                                 aggregation_equality>>
    _cache;
};

}  // namespace detail
}  // namespace cudf
