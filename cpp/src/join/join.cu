/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <join/hash_join.cuh>
#include <join/join_common_utils.hpp>
#include <join/nested_loop_join.cuh>

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned

  // now rebuild the table views with the updated ones
  auto const left  = matched.second.front();
  auto const right = matched.second.back();

  // For `inner_join`, we can freely choose either the `left` or `right` table to use for
  // building/probing the hash map. Because building is typically more expensive than probing, we
  // build the hash map from the smaller table.
  if (right.num_rows() > left.num_rows()) {
    cudf::hash_join hj_obj(left, compare_nulls, stream);
    auto result = hj_obj.inner_join(right, compare_nulls, std::nullopt, stream, mr);
    return std::make_pair(std::move(result.second), std::move(result.first));
  } else {
    cudf::hash_join hj_obj(right, compare_nulls, stream);
    return hj_obj.inner_join(left, compare_nulls, std::nullopt, stream, mr);
  }
}

std::unique_ptr<table> inner_join(table_view const& left_input,
                                  table_view const& right_input,
                                  std::vector<size_type> const& left_on,
                                  std::vector<size_type> const& right_on,
                                  null_equality compare_nulls,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input.select(left_on), right_input.select(right_on)},
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned

  // now rebuild the table views with the updated ones
  auto const left  = scatter_columns(matched.second.front(), left_on, left_input);
  auto const right = scatter_columns(matched.second.back(), right_on, right_input);

  auto join_indices = inner_join(left.select(left_on), right.select(right_on), compare_nulls, mr);
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      join_indices.first->begin(),
                                                      join_indices.first->end(),
                                                      out_of_bounds_policy::DONT_CHECK,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       join_indices.second->begin(),
                                                       join_indices.second->end(),
                                                       out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr);
  return combine_table_pair(std::move(left_result), std::move(right_result));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},  // these should match
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = matched.second.front();
  table_view const right = matched.second.back();

  cudf::hash_join hj_obj(right, compare_nulls, stream);
  return hj_obj.left_join(left, compare_nulls, std::nullopt, stream, mr);
}

std::unique_ptr<table> left_join(table_view const& left_input,
                                 table_view const& right_input,
                                 std::vector<size_type> const& left_on,
                                 std::vector<size_type> const& right_on,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input.select(left_on), right_input.select(right_on)},  // these should match
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = scatter_columns(matched.second.front(), left_on, left_input);
  table_view const right = scatter_columns(matched.second.back(), right_on, right_input);

  auto join_indices = left_join(left.select(left_on), right.select(right_on), compare_nulls);

  if ((left_on.empty() || right_on.empty()) ||
      is_trivial_join(left, right, cudf::detail::join_kind::LEFT_JOIN)) {
    auto probe_build_pair = get_empty_joined_table(left, right);
    return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                            std::move(probe_build_pair.second));
  }
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      join_indices.first->begin(),
                                                      join_indices.first->end(),
                                                      out_of_bounds_policy::NULLIFY,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       join_indices.second->begin(),
                                                       join_indices.second->end(),
                                                       out_of_bounds_policy::NULLIFY,
                                                       stream,
                                                       mr);
  return combine_table_pair(std::move(left_result), std::move(right_result));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},  // these should match
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = matched.second.front();
  table_view const right = matched.second.back();

  cudf::hash_join hj_obj(right, compare_nulls, stream);
  return hj_obj.full_join(left, compare_nulls, std::nullopt, stream, mr);
}

std::unique_ptr<table> full_join(table_view const& left_input,
                                 table_view const& right_input,
                                 std::vector<size_type> const& left_on,
                                 std::vector<size_type> const& right_on,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input.select(left_on), right_input.select(right_on)},  // these should match
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = scatter_columns(matched.second.front(), left_on, left_input);
  table_view const right = scatter_columns(matched.second.back(), right_on, right_input);

  auto join_indices = full_join(left.select(left_on), right.select(right_on), compare_nulls);

  if ((left_on.empty() || right_on.empty()) ||
      is_trivial_join(left, right, cudf::detail::join_kind::FULL_JOIN)) {
    auto probe_build_pair = get_empty_joined_table(left, right);
    return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                            std::move(probe_build_pair.second));
  }
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      join_indices.first->begin(),
                                                      join_indices.first->end(),
                                                      out_of_bounds_policy::NULLIFY,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       join_indices.second->begin(),
                                                       join_indices.second->end(),
                                                       out_of_bounds_policy::NULLIFY,
                                                       stream,
                                                       mr);
  return combine_table_pair(std::move(left_result), std::move(right_result));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
predicate_join(table_view left,
               table_view right,
               ast::expression binary_pred,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return get_base_nested_loop_predicate_join_indices(
    left, right, false, join_kind::INNER_JOIN, binary_pred, null_equality::EQUAL, stream);
}

}  // namespace detail

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : impl{std::make_unique<const hash_join::hash_join_impl>(build, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& probe,
                      null_equality compare_nulls,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr) const
{
  return impl->inner_join(probe, compare_nulls, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& probe,
                     null_equality compare_nulls,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const
{
  return impl->left_join(probe, compare_nulls, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& probe,
                     null_equality compare_nulls,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const
{
  return impl->full_join(probe, compare_nulls, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& probe,
                                       null_equality compare_nulls,
                                       rmm::cuda_stream_view stream) const
{
  return impl->inner_join_size(probe, compare_nulls, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& probe,
                                      null_equality compare_nulls,
                                      rmm::cuda_stream_view stream) const
{
  return impl->left_join_size(probe, compare_nulls, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& probe,
                                      null_equality compare_nulls,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr) const
{
  return impl->full_join_size(probe, compare_nulls, stream, mr);
}

// external APIs

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left,
           table_view const& right,
           null_equality compare_nulls,
           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::inner_join(left, right, compare_nulls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> inner_join(table_view const& left,
                                  table_view const& right,
                                  std::vector<size_type> const& left_on,
                                  std::vector<size_type> const& right_on,
                                  null_equality compare_nulls,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::inner_join(
    left, right, left_on, right_on, compare_nulls, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(table_view const& left,
          table_view const& right,
          null_equality compare_nulls,
          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_join(left, right, compare_nulls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> left_join(table_view const& left,
                                 table_view const& right,
                                 std::vector<size_type> const& left_on,
                                 std::vector<size_type> const& right_on,
                                 null_equality compare_nulls,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_join(
    left, right, left_on, right_on, compare_nulls, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(table_view const& left,
          table_view const& right,
          null_equality compare_nulls,
          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::full_join(left, right, compare_nulls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<table> full_join(table_view const& left,
                                 table_view const& right,
                                 std::vector<size_type> const& left_on,
                                 std::vector<size_type> const& right_on,
                                 null_equality compare_nulls,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::full_join(
    left, right, left_on, right_on, compare_nulls, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
predicate_join(table_view left,
               table_view right,
               ast::expression binary_pred,
               rmm::mr::device_memory_resource* mr)
{
  return detail::predicate_join(left, right, binary_pred, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
