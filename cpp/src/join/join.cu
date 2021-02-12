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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> inner_join(
  table_view const& left_input,
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

  // For `inner_join`, we can freely choose either the `left` or `right` table to use for
  // building/probing the hash map. Because building is typically more expensive than probing, we
  // build the hash map from the smaller table.
  if (right.num_rows() > left.num_rows()) {
    cudf::hash_join hj_obj(left, left_on, compare_nulls, stream);
    return hj_obj.inner_join(right, right_on, compare_nulls, stream, mr);
  } else {
    cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
    return hj_obj.inner_join(left, left_on, compare_nulls, stream, mr);
  }
}

std::unique_ptr<table> inner_join(
  table_view const& left_input,
  table_view const& right_input,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
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

  // For `inner_join`, we can freely choose either the `left` or `right` table to use for
  // building/probing the hash map. Because building is typically more expensive than probing, we
  // build the hash map from the smaller table.
  if (right.num_rows() > left.num_rows()) {
    cudf::hash_join hj_obj(left, left_on, compare_nulls, stream);
    auto join_indices = hj_obj.inner_join(right, right_on, compare_nulls, stream, mr);

    auto actual_columns_in_common = columns_in_common;
    std::for_each(actual_columns_in_common.begin(), actual_columns_in_common.end(), [](auto& pair) {
      std::swap(pair.first, pair.second);
    });

    if (is_trivial_join(left, right, left_on, right_on, cudf::detail::join_kind::INNER_JOIN)) {
      auto probe_build_pair = get_empty_joined_table(
        right, left, actual_columns_in_common, cudf::hash_join::common_columns_output_side::BUILD);
      return cudf::detail::combine_table_pair(std::move(probe_build_pair.second),
                                              std::move(probe_build_pair.first));
    }

    auto join_indices_view = std::make_pair<cudf::column_view, cudf::column_view>(
      join_indices.first->view(), join_indices.second->view());

    auto probe_build_pair = construct_join_output_df<cudf::detail::join_kind::INNER_JOIN>(
      right,
      left,
      join_indices_view,
      actual_columns_in_common,
      cudf::hash_join::common_columns_output_side::BUILD,
      stream,
      mr);

    return cudf::detail::combine_table_pair(std::move(probe_build_pair.second),
                                            std::move(probe_build_pair.first));
  } else {
    cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
    auto join_indices = hj_obj.inner_join(left, left_on, compare_nulls, stream, mr);

    if (is_trivial_join(left, right, left_on, right_on, cudf::detail::join_kind::INNER_JOIN)) {
      auto probe_build_pair = get_empty_joined_table(
        left, right, columns_in_common, cudf::hash_join::common_columns_output_side::PROBE);
      return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                              std::move(probe_build_pair.second));
    }

    auto join_indices_view = std::make_pair<cudf::column_view, cudf::column_view>(
      join_indices.first->view(), join_indices.second->view());

    auto probe_build_pair = construct_join_output_df<cudf::detail::join_kind::INNER_JOIN>(
      left,
      right,
      join_indices_view,
      columns_in_common,
      cudf::hash_join::common_columns_output_side::PROBE,
      stream,
      mr);

    return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                            std::move(probe_build_pair.second));
  }
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> left_join(
  table_view const& left_input,
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

  cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
  return hj_obj.left_join(left, left_on, compare_nulls, stream, mr);
}

std::unique_ptr<table> left_join(
  table_view const& left_input,
  table_view const& right_input,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
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

  cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
  auto join_indices = hj_obj.left_join(left, left_on, compare_nulls, stream, mr);

  if (is_trivial_join(left, right, left_on, right_on, cudf::detail::join_kind::LEFT_JOIN)) {
    auto probe_build_pair = get_empty_joined_table(
      left, right, columns_in_common, cudf::hash_join::common_columns_output_side::PROBE);
    return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                            std::move(probe_build_pair.second));
  }

  auto join_indices_view = std::make_pair<cudf::column_view, cudf::column_view>(
    join_indices.first->view(), join_indices.second->view());

  auto probe_build_pair = construct_join_output_df<cudf::detail::join_kind::LEFT_JOIN>(
    left,
    right,
    join_indices_view,
    columns_in_common,
    cudf::hash_join::common_columns_output_side::PROBE,
    stream,
    mr);

  return combine_table_pair(std::move(probe_build_pair.first), std::move(probe_build_pair.second));
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> full_join(
  table_view const& left_input,
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

  cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
  return hj_obj.full_join(left, left_on, compare_nulls, stream, mr);
}

std::unique_ptr<table> full_join(
  table_view const& left_input,
  table_view const& right_input,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
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

  cudf::hash_join hj_obj(right, right_on, compare_nulls, stream);
  auto join_indices = hj_obj.full_join(left, left_on, compare_nulls, stream, mr);

  if (is_trivial_join(left, right, left_on, right_on, cudf::detail::join_kind::FULL_JOIN)) {
    auto probe_build_pair = get_empty_joined_table(
      left, right, columns_in_common, cudf::hash_join::common_columns_output_side::PROBE);
    return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                            std::move(probe_build_pair.second));
  }

  auto join_indices_view = std::make_pair<cudf::column_view, cudf::column_view>(
    join_indices.first->view(), join_indices.second->view());

  auto probe_build_pair = construct_join_output_df<cudf::detail::join_kind::FULL_JOIN>(
    left,
    right,
    join_indices_view,
    columns_in_common,
    cudf::hash_join::common_columns_output_side::PROBE,
    stream,
    mr);

  return combine_table_pair(std::move(probe_build_pair.first), std::move(probe_build_pair.second));
}

}  // namespace detail

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     std::vector<size_type> const& build_on,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : impl{std::make_unique<const hash_join::hash_join_impl>(build, build_on, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> hash_join::inner_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->inner_join(probe, probe_on, compare_nulls, stream, mr);
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> hash_join::left_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->left_join(probe, probe_on, compare_nulls, stream, mr);
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> hash_join::full_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->full_join(probe, probe_on, compare_nulls, stream, mr);
}

// external APIs

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> inner_join(
  table_view const& left,
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

std::unique_ptr<table> inner_join(
  table_view const& left,
  table_view const& right,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::inner_join(
    left, right, left_on, right_on, columns_in_common, compare_nulls, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> left_join(
  table_view const& left,
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

std::unique_ptr<table> left_join(
  table_view const& left,
  table_view const& right,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_join(
    left, right, left_on, right_on, columns_in_common, compare_nulls, rmm::cuda_stream_default, mr);
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> full_join(
  table_view const& left,
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

std::unique_ptr<table> full_join(
  table_view const& left,
  table_view const& right,
  std::vector<size_type> const& left_on,
  std::vector<size_type> const& right_on,
  std::vector<std::pair<size_type, size_type>> const& columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::full_join(
    left, right, left_on, right_on, columns_in_common, compare_nulls, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
