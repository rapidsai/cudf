/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/detail/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_map.cuh>

namespace cudf::detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

/**
 * @brief Functor to remap hash values to new values if they are equal to the specified sentinel
 * value.
 */
template <typename Hasher>
struct remap_sentinel_hasher {
  remap_sentinel_hasher(Hasher&& row_hasher, hash_value_type const sentinel)
    : _hasher{std::move(row_hasher)}, _sentinel(sentinel)
  {
  }

  template <typename T>
  __device__ inline auto operator()(T const idx) const noexcept
  {
    return remap_sentinel_hash(_hasher(static_cast<size_type>(idx)), _sentinel);
  }

 private:
  Hasher _hasher;
  hash_value_type const _sentinel;
};

/**
 * @brief A comparator adapter so that the underlying self comparator can work with strong index
 * types.
 */
template <typename Comparator>
struct strong_index_self_comparator_adapter {
  strong_index_self_comparator_adapter(Comparator const& comparator) : _comparator{comparator} {}

  template <typename T>
  __device__ inline auto operator()(T const lhs_index, T const rhs_index) const noexcept
  {
    return _comparator(static_cast<size_type>(lhs_index), static_cast<size_type>(rhs_index));
  }

 private:
  Comparator const _comparator;
};

/**
 * @brief Invoke an `operator()` template with a row equality comparator based on the specified
 * `compare_nans` parameter.
 *
 * @param compare_nans The flag to specify whether NaNs should be compared equal or not
 * @param func The input functor to invoke
 */
template <typename Func>
void dispatch_nan_comparator(nan_equality compare_nans, Func&& func)
{
  if (compare_nans == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    func(nan_equal_comparator{});
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    func(nan_unequal_comparator{});
  }
}

}  // namespace

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  using static_map = cuco::static_map<lhs_index_type,
                                      size_type,
                                      cuda::thread_scope_device,
                                      rmm::mr::stream_allocator_adaptor<default_allocator<char>>>;

  auto map = static_map(compute_hash_table_size(haystack.num_rows()),
                        cuco::sentinel::empty_key{std::numeric_limits<lhs_index_type>::max()},
                        cuco::sentinel::empty_value{detail::JoinNoneValue},
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value());

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  // Insert all row hash values and indices of the haystack table.
  {
    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      [] __device__(auto const idx) { return cuco::make_pair(lhs_index_type{idx}, 0); });

    auto const hasher = cudf::experimental::row::hash::row_hasher(haystack, stream);
    auto const d_hasher =
      remap_sentinel_hasher(hasher.device_hasher(nullate::DYNAMIC{has_any_nulls}),
                            static_cast<hash_value_type>(map.get_empty_key_sentinel()));

    auto const comparator = cudf::experimental::row::equality::self_comparator(haystack, stream);

    // If the haystack table has nulls but they are compared unequal, don't insert them.
    // Otherwise, it was known to cause performance issue:
    // - https://github.com/rapidsai/cudf/pull/6943
    // - https://github.com/rapidsai/cudf/pull/8277
    if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      // Collect all nullable columns at all levels from the haystack table.
      auto const haystack_nullable_columns = get_nullable_columns(haystack);
      CUDF_EXPECTS(haystack_nullable_columns.size() > 0,
                   "Haystack table has nulls thus it should have nullable columns.");

      // If there are more than one nullable column, we compute bitmask_and of their null masks.
      // Otherwise, we have only one nullable column and can use its null mask directly.
      auto const row_bitmask =
        haystack_nullable_columns.size() > 1
          ? cudf::detail::bitmask_and(table_view{haystack_nullable_columns}, stream).first
          : rmm::device_buffer{0, stream};
      auto const row_bitmask_ptr = haystack_nullable_columns.size() > 1
                                     ? static_cast<bitmask_type const*>(row_bitmask.data())
                                     : haystack_nullable_columns.front().null_mask();

      // Insert only rows that do not have any null at any level.
      auto const insert_map = [&](auto const value_comp) {
        auto const d_eqcomp = strong_index_self_comparator_adapter{
          comparator.equal_to(nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, value_comp)};
        map.insert_if(haystack_it,
                      haystack_it + haystack.num_rows(),
                      thrust::counting_iterator<size_type>(0),  // stencil
                      row_is_valid{row_bitmask_ptr},
                      d_hasher,
                      d_eqcomp,
                      stream.value());
      };

      dispatch_nan_comparator(compare_nans, insert_map);

    } else {  // haystack_doesn't_have_nulls || compare_nulls == null_equality::EQUAL
      auto const insert_map = [&](auto const value_comp) {
        auto const d_eqcomp = strong_index_self_comparator_adapter{
          comparator.equal_to(nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, value_comp)};
        map.insert(
          haystack_it, haystack_it + haystack.num_rows(), d_hasher, d_eqcomp, stream.value());
      };

      dispatch_nan_comparator(compare_nans, insert_map);
    }
  }

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Check existence for each row of the needles table in the haystack table.
  {
    auto const needles_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [] __device__(auto const idx) { return rhs_index_type{idx}; });

    auto const hasher = cudf::experimental::row::hash::row_hasher(needles, stream);
    auto const d_hasher =
      remap_sentinel_hasher(hasher.device_hasher(nullate::DYNAMIC{has_any_nulls}),
                            static_cast<hash_value_type>(map.get_empty_key_sentinel()));

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(haystack, needles, stream);

    auto const check_contains = [&](auto const value_comp) {
      auto const d_eqcomp =
        comparator.equal_to(nullate::DYNAMIC{has_any_nulls}, compare_nulls, value_comp);
      map.contains(needles_it,
                   needles_it + needles.num_rows(),
                   contained.begin(),
                   d_hasher,
                   d_eqcomp,
                   stream.value());
    };

    dispatch_nan_comparator(compare_nans, check_contains);
  }

  return contained;
}

}  // namespace cudf::detail
