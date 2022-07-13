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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cuco/static_multimap.cuh>

namespace cudf::detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

}  // namespace

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  // Use a hash map with key type is row hash values and map value type is `lhs_index_type` to store
  // all row indices of the haystack table.
  using static_multimap =
    cuco::static_multimap<hash_value_type,
                          lhs_index_type,
                          cuda::thread_scope_device,
                          rmm::mr::stream_allocator_adaptor<default_allocator<char>>,
                          cuco::double_hashing<detail::DEFAULT_JOIN_CG_SIZE, hash_type, hash_type>>;

  auto map = static_multimap(compute_hash_table_size(haystack.num_rows()),
                             cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
                             cuco::sentinel::empty_value{lhs_index_type{detail::JoinNoneValue}},
                             stream.value(),
                             detail::hash_table_allocator_type{default_allocator<char>{}, stream});

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);

  // Insert all row hash values and indices of the haystack table.
  {
    auto const hasher   = cudf::experimental::row::hash::row_hasher(haystack, stream);
    auto const d_hasher = hasher.device_hasher(nullate::DYNAMIC{haystack_has_nulls});

    using make_pair_fn = make_pair_function<decltype(d_hasher), lhs_index_type>;

    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, make_pair_fn{d_hasher, map.get_empty_key_sentinel()});

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
      map.insert_if(haystack_it,
                    haystack_it + haystack.num_rows(),
                    thrust::counting_iterator<size_type>(0),  // stencil
                    row_is_valid{row_bitmask_ptr},
                    stream.value());
    } else {
      map.insert(haystack_it, haystack_it + haystack.num_rows(), stream.value());
    }
  }

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Check existence for each row of the needles table in the haystack table.
  {
    auto const hasher   = cudf::experimental::row::hash::row_hasher(needles, stream);
    auto const d_hasher = hasher.device_hasher(nullate::DYNAMIC{needles_has_nulls});

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(haystack, needles, stream);

    using make_pair_fn = make_pair_function<decltype(d_hasher), rhs_index_type>;

    auto const needles_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, make_pair_fn{d_hasher, map.get_empty_key_sentinel()});

    auto const check_contains = [&](auto const value_comp) {
      auto const d_eqcomp = comparator.equal_to(
        nullate::DYNAMIC{needles_has_nulls || haystack_has_nulls}, compare_nulls, value_comp);
      map.pair_contains(needles_it,
                        needles_it + needles.num_rows(),
                        contained.begin(),
                        pair_equality{d_eqcomp},
                        stream.value());
    };

    if (compare_nans == nan_equality::ALL_EQUAL) {
      using nan_equal_comparator =
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
      check_contains(nan_equal_comparator{});
    } else {
      using nan_unequal_comparator =
        cudf::experimental::row::equality::physical_equality_comparator;
      check_contains(nan_unequal_comparator{});
    }
  }

  return contained;
}

}  // namespace cudf::detail
