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
namespace experimental {
namespace groupby {
namespace detail {

struct aggregation_equality {
  struct typed_aggregation_equality
  {
    std::shared_ptr<aggregation> const& lhs;
    std::shared_ptr<aggregation> const& rhs;

    template <aggregation::Kind k>
    bool operator()() const {
      using agg_type = experimental::detail::kind_to_type<k>;
      auto typed_lhs = static_cast<agg_type const*>(lhs.get());
      auto typed_rhs = static_cast<agg_type const*>(rhs.get());
      return *typed_lhs == *typed_rhs;
    }
  };

  bool operator()(std::shared_ptr<aggregation> const& lhs,
                  std::shared_ptr<aggregation> const& rhs) const
  {
    if (lhs->kind != rhs->kind)
      return false;
    
    return experimental::detail::aggregation_dispatcher(lhs->kind,
              typed_aggregation_equality{lhs, rhs});
  }
};

struct aggregation_hash {
  size_t operator()(std::shared_ptr<aggregation> const& key) const noexcept {
    if (key) {
      return key->kind;
    } else {
      return 0;
    }
  }
};

class result_cache {
 public:
  result_cache() = delete;
  ~result_cache() = default;
  result_cache(result_cache const&) = delete;
  result_cache& operator=(const result_cache& other) = delete;

  result_cache(size_t num_columns)
  : _cache(num_columns)
  {}

  bool has_result(size_t col_idx, std::unique_ptr<aggregation> const& agg);

  void add_result(size_t col_idx, std::unique_ptr<aggregation> const& agg,
                  std::unique_ptr<column>&& col);

  column_view
  get_result(size_t col_idx, std::unique_ptr<aggregation> const& agg);

  std::unique_ptr<column>
  release_result(size_t col_idx, std::unique_ptr<aggregation> const& agg);

 private:
  std::vector<
    std::unordered_map<
      std::shared_ptr<aggregation>,
      std::unique_ptr<column>,
      aggregation_hash,
      aggregation_equality
    >
  > _cache;
};


}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
