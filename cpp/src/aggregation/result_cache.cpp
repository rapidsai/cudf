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

#include <cudf/detail/aggregation/result_cache.hpp>

namespace cudf {
namespace detail {

bool result_cache::has_result(size_t col_idx, aggregation const& agg) const
{
  if (col_idx > _cache.size()) return false;

  auto result_it = _cache[col_idx].find(agg);

  return (result_it != _cache[col_idx].end());
}

void result_cache::add_result(size_t col_idx, aggregation const& agg, std::unique_ptr<column>&& col)
{
  // We can't guarantee that agg will outlive the cache, so we need to take ownership of a copy.
  // To allow lookup by reference, make the key a reference and keep the owner in the value pair.
  auto owned_agg  = agg.clone();
  auto const& key = *owned_agg;
  auto value      = std::make_pair(std::move(owned_agg), std::move(col));
  _cache[col_idx].emplace(key, std::move(value));
}

column_view result_cache::get_result(size_t col_idx, aggregation const& agg) const
{
  CUDF_EXPECTS(has_result(col_idx, agg), "Result does not exist in cache");

  auto result_it = _cache[col_idx].find(agg);
  return result_it->second.second->view();
}

std::unique_ptr<column> result_cache::release_result(size_t col_idx, aggregation const& agg)
{
  CUDF_EXPECTS(has_result(col_idx, agg), "Result does not exist in cache");

  auto result_it = _cache[col_idx].extract(agg);
  return std::move(result_it.mapped().second);
}

}  // namespace detail
}  // namespace cudf
