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

#include <cudf/column/column_factories.hpp>
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
#include <thrust/tuple.h>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief Device functor to create a pair of hash value and index for a given row.
 */
struct make_pair_fn {
  __device__ __forceinline__ cudf::detail::pair_type operator()(size_type i) const noexcept
  {
    // The value is irrelevant since we only ever use the hash map to check for
    // membership of a particular row index.
    return cuco::make_pair(static_cast<hash_value_type>(i), 0);
  }
};

}  // namespace

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
  auto right_flattened_tables = structs::detail::flatten_nested_columns(
    right_keys, {}, {}, structs::detail::column_nullability::FORCE);
  auto left_flattened_tables = structs::detail::flatten_nested_columns(
    left_keys, {}, {}, structs::detail::column_nullability::FORCE);

  auto right_flattened_keys = right_flattened_tables.flattened_columns();
  auto left_flattened_keys  = left_flattened_tables.flattened_columns();

  // Create hash table.
  semi_map_type hash_table{compute_hash_table_size(right_num_rows),
                           cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
                           cuco::sentinel::empty_value{cudf::detail::JoinNoneValue},
                           hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  // Create hash table containing all keys found in right table
  auto right_rows_d      = table_device_view::create(right_flattened_keys, stream);
  auto const right_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(right_flattened_keys)};
  row_hash const hash_build{right_nulls, *right_rows_d};
  row_equality equality_build{right_nulls, *right_rows_d, *right_rows_d, compare_nulls};
  make_pair_fn pair_func_build{};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func_build);

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
  auto left_rows_d      = table_device_view::create(left_flattened_keys, stream);
  auto const left_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(left_flattened_keys)};
  row_hash hash_probe{left_nulls, *left_rows_d};
  // Note: This equality comparator violates symmetry of equality and is
  // therefore relying on the implementation detail of the order in which its
  // operator is invoked. If cuco makes no promises about the order of
  // invocation this seems a bit unsafe.
  row_equality equality_probe{left_nulls, *right_rows_d, *left_rows_d, compare_nulls};

  // For semi join we want contains to be true, for anti join we want contains to be false
  bool const join_type_boolean = (kind == join_kind::LEFT_SEMI_JOIN);

  auto hash_table_view = hash_table.get_device_view();

  auto gather_map =
    std::make_unique<rmm::device_uvector<cudf::size_type>>(left_num_rows, stream, mr);

  rmm::device_uvector<bool> flagged(left_num_rows, stream, mr);
  auto flagged_d = flagged.data();

  auto counting_iter = thrust::counting_iterator<size_type>(0);
  thrust::for_each(
    rmm::exec_policy(stream),
    counting_iter,
    counting_iter + left_num_rows,
    [flagged_d, hash_table_view, join_type_boolean, hash_probe, equality_probe] __device__(
      const size_type idx) {
      flagged_d[idx] =
        hash_table_view.contains(idx, hash_probe, equality_probe) == join_type_boolean;
    });

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    counting_iter,
                    counting_iter + left_num_rows,
                    gather_map->begin(),
                    [flagged_d] __device__(size_type const idx) { return flagged_d[idx]; });

  auto join_size = thrust::distance(gather_map->begin(), gather_map_end);
  gather_map->resize(join_size, stream);
  return gather_map;
}

/**
 * @brief  Performs a left semi or anti join on the specified columns of two
 * tables (left, right)
 *
 * The semi and anti joins only return data from the left table. A left semi join
 * returns rows that exist in the right table, a left anti join returns rows
 * that do not exist in the right table.
 *
 * The basic approach is to create a hash table containing the contents of the right
 * table and then select only rows that exist (or don't exist) to be included in
 * the return set.
 *
 * @throws cudf::logic_error if number of columns in either `left` or `right` table is 0
 * @throws cudf::logic_error if number of returned columns is 0
 * @throws cudf::logic_error if number of elements in `right_on` and `left_on` are not equal
 *
 * @param kind          Indicates whether to do LEFT_SEMI_JOIN or LEFT_ANTI_JOIN
 * @param left          The left table
 * @param right         The right table
 * @param left_on       The column indices from `left` to join on.
 *                      The column from `left` indicated by `left_on[i]`
 *                      will be compared against the column from `right`
 *                      indicated by `right_on[i]`.
 * @param right_on      The column indices from `right` to join on.
 *                      The column from `right` indicated by `right_on[i]`
 *                      will be compared against the column from `left`
 *                      indicated by `left_on[i]`.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream        CUDA stream used for device memory operations and kernel launches.
 * @param mr            Device memory resource to used to allocate the returned table
 *
 * @returns             Result of joining `left` and `right` tables on the columns
 *                      specified by `left_on` and `right_on`.
 */
std::unique_ptr<cudf::table> left_semi_anti_join(
  join_kind const kind,
  cudf::table_view const& left,
  cudf::table_view const& right,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(left_on.size() == right_on.size(), "Mismatch in number of columns to be joined on");

  if ((left_on.empty() || right_on.empty()) || is_trivial_join(left, right, kind)) {
    return empty_like(left);
  }

  if ((join_kind::LEFT_ANTI_JOIN == kind) && (0 == right.num_rows())) {
    // Everything matches, just copy the proper columns from the left table
    return std::make_unique<table>(left, stream, mr);
  }

  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left.select(left_on), right.select(right_on)},
    stream,
    rmm::mr::get_current_device_resource());  // temporary objects returned

  auto const left_selected  = matched.second.front();
  auto const right_selected = matched.second.back();

  auto gather_vector =
    left_semi_anti_join(kind, left_selected, right_selected, compare_nulls, stream);

  // wrapping the device vector with a column view allows calling the non-iterator
  // version of detail::gather, improving compile time by 10% and reducing the
  // object file size by 2.2x without affecting performance
  auto gather_map = column_view(data_type{type_id::INT32},
                                static_cast<size_type>(gather_vector->size()),
                                gather_vector->data(),
                                nullptr,
                                0);

  auto const left_updated = scatter_columns(left_selected, left_on, left);
  return cudf::detail::gather(left_updated,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> left_semi_join(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<cudf::size_type> const& left_on,
                                            std::vector<cudf::size_type> const& right_on,
                                            null_equality compare_nulls,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(detail::join_kind::LEFT_SEMI_JOIN,
                                     left,
                                     right,
                                     left_on,
                                     right_on,
                                     compare_nulls,
                                     cudf::default_stream_value,
                                     mr);
}

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

std::unique_ptr<cudf::table> left_anti_join(cudf::table_view const& left,
                                            cudf::table_view const& right,
                                            std::vector<cudf::size_type> const& left_on,
                                            std::vector<cudf::size_type> const& right_on,
                                            null_equality compare_nulls,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(detail::join_kind::LEFT_ANTI_JOIN,
                                     left,
                                     right,
                                     left_on,
                                     right_on,
                                     compare_nulls,
                                     cudf::default_stream_value,
                                     mr);
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
