/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>

#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace detail {

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_anti_join(
  join_kind const kind,
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(0 != left_keys.num_columns(), "Left table is empty");
  CUDF_EXPECTS(0 != right_keys.num_columns(), "Right table is empty");

  if (is_trivial_join(left_keys, right_keys, kind)) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if ((join_kind::LEFT_ANTI_JOIN == kind) && (0 == right_keys.num_rows())) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(left_keys.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy(stream), result->begin(), result->end());
    return result;
  }

  auto const left_num_rows  = left_keys.num_rows();
  auto const right_num_rows = right_keys.num_rows();

  // flatten structs for the right and left and use that for the hash table
  auto const right_flattened_tables = structs::detail::flatten_nested_columns(
    right_keys, {}, {}, structs::detail::column_nullability::FORCE);
  auto const left_flattened_tables = structs::detail::flatten_nested_columns(
    left_keys, {}, {}, structs::detail::column_nullability::FORCE);
  auto const right_flattened_keys = right_flattened_tables.flattened_columns();
  auto const left_flattened_keys  = left_flattened_tables.flattened_columns();

  // Create hash table.
  semi_map_type hash_table{compute_hash_table_size(right_num_rows),
                           cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
                           cuco::sentinel::empty_value{cudf::detail::JoinNoneValue},
                           hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  // Create hash table containing all keys found in right table
  auto const right_rows_d = table_device_view::create(right_flattened_keys, stream);
  auto const right_nulls  = cudf::nullate::DYNAMIC{cudf::has_nulls(right_flattened_keys)};
  row_hash const hash_build{right_nulls, *right_rows_d};
  row_equality equality_build{right_nulls, *right_rows_d, *right_rows_d, compare_nulls};

  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(auto const i) { return cuco::make_pair(static_cast<hash_value_type>(i), 0); });

  // skip rows that are null here.
  if ((compare_nulls == null_equality::EQUAL) or (not nullable(right_keys))) {
    hash_table.insert(iter, iter + right_num_rows, hash_build, equality_build, stream.value());
  } else {
    thrust::counting_iterator<size_type> stencil(0);
    auto const [row_bitmask, _] = cudf::detail::bitmask_and(right_flattened_keys, stream);
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    hash_table.insert_if(
      iter, iter + right_num_rows, stencil, pred, hash_build, equality_build, stream.value());
  }

  // Now we have a hash table, we need to iterate over the rows of the left table
  // and check to see if they are contained in the hash table
  auto const left_rows_d = table_device_view::create(left_flattened_keys, stream);
  auto const left_nulls  = cudf::nullate::DYNAMIC{cudf::has_nulls(left_flattened_keys)};
  row_hash hash_probe{left_nulls, *left_rows_d};
  // Note: This equality comparator violates symmetry of equality and is
  // therefore relying on the implementation detail of the order in which its
  // operator is invoked. If cuco makes no promises about the order of
  // invocation this seems a bit unsafe.
  row_equality equality_probe{left_nulls, *right_rows_d, *left_rows_d, compare_nulls};

  // For semi join we want contains to be true, for anti join we want contains to be false
  bool const join_type_boolean = (kind == join_kind::LEFT_SEMI_JOIN);

  auto gather_map =
    std::make_unique<rmm::device_uvector<cudf::size_type>>(left_num_rows, stream, mr);

  rmm::device_uvector<bool> flagged(left_num_rows, stream, mr);
  auto flagged_d = flagged.data();

  auto hash_table_view = hash_table.get_device_view();
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(left_num_rows),
    [flagged_d, hash_table_view, join_type_boolean, hash_probe, equality_probe] __device__(
      const size_type idx) {
      flagged_d[idx] =
        hash_table_view.contains(idx, hash_probe, equality_probe) == join_type_boolean;
    });

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(left_num_rows),
                    gather_map->begin(),
                    [flagged_d] __device__(size_type const idx) { return flagged_d[idx]; });

  gather_map->resize(thrust::distance(gather_map->begin(), gather_map_end), stream);
  return gather_map;
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(
    detail::join_kind::LEFT_SEMI_JOIN, left, right, compare_nulls, cudf::default_stream_value, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_anti_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(
    detail::join_kind::LEFT_ANTI_JOIN, left, right, compare_nulls, cudf::default_stream_value, mr);
}

}  // namespace cudf
