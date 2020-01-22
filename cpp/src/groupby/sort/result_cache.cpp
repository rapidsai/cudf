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

#include "result_cache.hpp"

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {

namespace {

struct typed_agg_copier
{
 private:
  std::unique_ptr<aggregation> const& agg;

 public:
  typed_agg_copier(std::unique_ptr<aggregation> const& agg) : agg(agg) {}

  template <aggregation::Kind k>
  std::shared_ptr<aggregation> operator()() {
    using agg_type = experimental::detail::kind_to_type<k>;
    auto typed_agg = static_cast<agg_type const*>(agg.get());
    aggregation* copy = new agg_type(*typed_agg);
    return std::shared_ptr<aggregation>(copy);
  }
};

std::shared_ptr<aggregation>
copy_to_shared_ptr(std::unique_ptr<aggregation> const& agg) {
  return experimental::detail::aggregation_dispatcher(agg->kind, 
                                  typed_agg_copier(agg));
}

} // namespace

bool result_cache::has_result(size_t col_idx, 
                              std::unique_ptr<aggregation> const& agg)
{
  if (col_idx < 0 or col_idx > _cache.size())
    return false;

  auto agg_copy = copy_to_shared_ptr(agg);
  auto result_it = _cache[col_idx].find(agg_copy);

  if (result_it != _cache[col_idx].end())
    return true;
  else
    return false;
}

void result_cache::add_result(size_t col_idx, 
                              std::unique_ptr<aggregation> const& agg,
                              std::unique_ptr<column>&& col)
{
  auto key = copy_to_shared_ptr(agg);
  column* col_ptr = col.release();
  _cache[col_idx].emplace(std::move(key), std::unique_ptr<column>(col_ptr));
}

column_view result_cache::get_result(size_t col_idx,
                                     std::unique_ptr<aggregation> const& agg)
{
  CUDF_EXPECTS(has_result(col_idx, agg), "Result does not exist in cache");

  auto key = copy_to_shared_ptr(agg);
  auto result_it = _cache[col_idx].find(key);
  return result_it->second->view();
}

std::unique_ptr<column> 
result_cache::release_result(size_t col_idx,
                             std::unique_ptr<aggregation> const& agg)
{
  CUDF_EXPECTS(has_result(col_idx, agg), "Result does not exist in cache");

  auto key = copy_to_shared_ptr(agg);

  // unordered_map.extract() is a c++17 feature so we do this:
  auto result_it = _cache[col_idx].find(key);
  column* raw_column = result_it->second.release();
  _cache[col_idx].erase(result_it);
  return std::unique_ptr<column>(raw_column);
}

}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
