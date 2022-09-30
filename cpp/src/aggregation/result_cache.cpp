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

bool result_cache::has_result(column_view const& input, aggregation const& agg) const
{
  return _cache.count({input, agg});
}

void result_cache::add_result(column_view const& input,
                              aggregation const& agg,
                              std::unique_ptr<column>&& col)
{
  // We can't guarantee that agg will outlive the cache, so we need to take ownership of a copy.
  // To allow lookup by reference, make the key a reference and keep the owner in the value pair.
  auto owned_agg  = agg.clone();
  auto const& key = *owned_agg;
  // try_emplace doesn't update/insert if already present
  _cache.try_emplace({input, key}, std::move(owned_agg), std::move(col));
}

column_view result_cache::get_result(column_view const& input, aggregation const& agg) const
{
  auto result_it = _cache.find({input, agg});
  CUDF_EXPECTS(result_it != _cache.end(), "Result does not exist in cache");
  return result_it->second.second->view();
}

std::unique_ptr<column> result_cache::release_result(column_view const& input,
                                                     aggregation const& agg)
{
  auto node = _cache.extract({input, agg});
  CUDF_EXPECTS(not node.empty(), "Result does not exist in cache");
  return std::move(node.mapped().second);
}

}  // namespace detail
}  // namespace cudf
