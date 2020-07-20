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
#include <cudf/table/table.hpp>

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include "cudf/types.hpp"
#include "hash_join.cuh"

namespace cudf {

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
  // The `right` table is always used for building the hash table. We want to build the hash table
  // on the smaller table. Thus, if `left` is smaller than `right`, swap `left/right`.
  if (right.num_rows() > left.num_rows()) {
    cudf::hash_join hj_obj(left, left_on);
    return hj_obj.inner_join(right,
                             right_on,
                             columns_in_common,
                             cudf::hash_join::probe_output_side::RIGHT,
                             compare_nulls,
                             mr);
  }
  cudf::hash_join hj_obj(right, right_on);
  return hj_obj.inner_join(
    left, left_on, columns_in_common, cudf::hash_join::probe_output_side::LEFT, compare_nulls, mr);
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
  cudf::hash_join hj_obj(right, right_on);
  return hj_obj.left_join(left, left_on, columns_in_common, compare_nulls, mr);
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
  cudf::hash_join hj_obj(right, right_on);
  return hj_obj.full_join(left, left_on, columns_in_common, compare_nulls, mr);
}

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build, std::vector<size_type> const& build_on)
  : impl{std::make_unique<const hash_join::hash_join_impl>(build, build_on)}
{
}

std::unique_ptr<cudf::table> hash_join::inner_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  probe_output_side probe_output_side,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->inner_join(probe, probe_on, columns_in_common, probe_output_side, compare_nulls, mr);
}

std::unique_ptr<cudf::table> hash_join::left_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->left_join(probe, probe_on, columns_in_common, compare_nulls, mr);
}

std::unique_ptr<cudf::table> hash_join::full_join(
  cudf::table_view const& probe,
  std::vector<size_type> const& probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr) const
{
  return impl->full_join(probe, probe_on, columns_in_common, compare_nulls, mr);
}

}  // namespace cudf
