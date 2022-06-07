/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include "join_common_utils.hpp"

#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
namespace {
std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const& probe, table_view const& build)
{
  std::unique_ptr<table> empty_probe = empty_like(probe);
  std::unique_ptr<table> empty_build = empty_like(build);
  return std::pair(std::move(empty_probe), std::move(empty_build));
}

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table>&& left,
                                                std::unique_ptr<cudf::table>&& right)
{
  auto joined_cols = left->release();
  auto right_cols  = right->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}
}  // namespace

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
    auto [right_result, left_result] = hj_obj.inner_join(right, std::nullopt, stream, mr);
    return std::pair(std::move(left_result), std::move(right_result));
  } else {
    cudf::hash_join hj_obj(right, compare_nulls, stream);
    return hj_obj.inner_join(left, std::nullopt, stream, mr);
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

  auto const [left_join_indices, right_join_indices] = cudf::detail::inner_join(
    left.select(left_on), right.select(right_on), compare_nulls, stream, mr);
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      left_join_indices->begin(),
                                                      left_join_indices->end(),
                                                      out_of_bounds_policy::DONT_CHECK,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       right_join_indices->begin(),
                                                       right_join_indices->end(),
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
  return hj_obj.left_join(left, std::nullopt, stream, mr);
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

  if ((left_on.empty() or right_on.empty()) or
      cudf::detail::is_trivial_join(left, right, cudf::detail::join_kind::LEFT_JOIN)) {
    auto [left_empty_table, right_empty_table] = get_empty_joined_table(left, right);
    return cudf::detail::combine_table_pair(std::move(left_empty_table),
                                            std::move(right_empty_table));
  }

  auto const [left_join_indices, right_join_indices] = cudf::detail::left_join(
    left.select(left_on), right.select(right_on), compare_nulls, stream, mr);
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      left_join_indices->begin(),
                                                      left_join_indices->end(),
                                                      out_of_bounds_policy::NULLIFY,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       right_join_indices->begin(),
                                                       right_join_indices->end(),
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
  return hj_obj.full_join(left, std::nullopt, stream, mr);
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

  if ((left_on.empty() or right_on.empty()) or
      cudf::detail::is_trivial_join(left, right, cudf::detail::join_kind::FULL_JOIN)) {
    auto [left_empty_table, right_empty_table] = get_empty_joined_table(left, right);
    return cudf::detail::combine_table_pair(std::move(left_empty_table),
                                            std::move(right_empty_table));
  }

  auto const [left_join_indices, right_join_indices] = cudf::detail::full_join(
    left.select(left_on), right.select(right_on), compare_nulls, stream, mr);
  std::unique_ptr<table> left_result  = detail::gather(left,
                                                      left_join_indices->begin(),
                                                      left_join_indices->end(),
                                                      out_of_bounds_policy::NULLIFY,
                                                      stream,
                                                      mr);
  std::unique_ptr<table> right_result = detail::gather(right,
                                                       right_join_indices->begin(),
                                                       right_join_indices->end(),
                                                       out_of_bounds_policy::NULLIFY,
                                                       stream,
                                                       mr);
  return combine_table_pair(std::move(left_result), std::move(right_result));
}
}  // namespace detail

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
}  // namespace cudf
