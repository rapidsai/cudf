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
#include <join/join_common_utils.hpp>

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

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 * @tparam T Type of row index, must be `size_type` or a strong index type.
 * @tparam Hasher The type of internal hasher to compute row hash.
 */
template <typename T, typename Hasher>
struct make_pair_fn {
  Hasher const hasher;
  hash_value_type const empty_key_sentinel;

  __device__ inline auto operator()(size_type const i) const
  {
    auto const hash_value = remap_sentinel_hash(hasher(i), empty_key_sentinel);
    return cuco::make_pair(hash_value, T{i});
  }
};

/**
 * @brief The functor to compare two rows using row hashes and row indices.
 *
 * @tparam Comparator The row comparator type to perform row equality comparison from row indices.
 */
template <typename Comparator>
struct pair_comparator_fn {
  Comparator const d_eqcomp;

  pair_comparator_fn(Comparator const d_eqcomp) : d_eqcomp{d_eqcomp} {}

  template <typename LHSPair, typename RHSPair>
  __device__ inline bool operator()(LHSPair const& lhs_hash_and_index,
                                    RHSPair const& rhs_hash_and_index) const
  {
    auto const& [lhs_hash, lhs_index] = lhs_hash_and_index;
    auto const& [rhs_hash, rhs_index] = rhs_hash_and_index;
    return lhs_hash == rhs_hash ? d_eqcomp(lhs_index, rhs_index) : false;
  }
};

/**
 * @brief The functor to accumulate all nullable columns at all nested levels from a given column.
 *
 * This is to avoid expensive materializing the bitmask into a real column when calling to
 * `structs::detail::flatten_nested_columns`.
 */
struct accumulate_nullable_columns {
  std::vector<column_view> result;

  accumulate_nullable_columns(table_view const& table)
  {
    for (auto const& col : table) {
      accumulate(col);
    }
  }

  auto release() { return std::move(result); }

 private:
  void accumulate(column_view const& col)
  {
    if (col.nullable()) { result.push_back(col); }
    for (auto it = col.child_begin(); it != col.child_end(); ++it) {
      if (it->size() == col.size()) { accumulate(*it); }
    }
  }
};

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

    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      make_pair_fn<lhs_index_type, decltype(d_hasher)>{d_hasher, map.get_empty_key_sentinel()});

    // If the haystack table has nulls but they are compared unequal, don't insert them.
    // Otherwise, it was known to cause performance issue:
    // - https://github.com/rapidsai/cudf/pull/6943
    // - https://github.com/rapidsai/cudf/pull/8277
    if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      // Gather all nullable columns at all levels from the right table.
      auto const nullable_columns = accumulate_nullable_columns{haystack}.release();
      CUDF_EXPECTS(nullable_columns.size() > 0,
                   "The haystack table has nulls but cannot collect any nullable column.");

      // If there is just a single nullable column, just use the column nullmask directly
      // to avoid launching a kernel for `bitmask_and`.
      auto const buff = [&] {
        if (nullable_columns.size() == 1) { return rmm::device_buffer{0, stream}; }
        return std::move(cudf::detail::bitmask_and(table_view{nullable_columns}, stream).first);
      }();
      auto const row_bitmask = buff.size() > 0 ? static_cast<bitmask_type const*>(buff.data())
                                               : nullable_columns.front().null_mask();

      // Insert only rows that do not have any nulls at any level.
      map.insert_if(haystack_it,
                    haystack_it + haystack.num_rows(),
                    thrust::counting_iterator<size_type>(0),  // stencil
                    row_is_valid{row_bitmask},                // pred
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

    auto const needles_it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      make_pair_fn<rhs_index_type, decltype(d_hasher)>{d_hasher, map.get_empty_key_sentinel()});

    auto const check_contains = [&](auto const value_comp) {
      auto const d_eqcomp = comparator.equal_to(
        nullate::DYNAMIC{needles_has_nulls || haystack_has_nulls}, compare_nulls, value_comp);
      map.pair_contains(needles_it,
                        needles_it + needles.num_rows(),
                        contained.begin(),
                        pair_comparator_fn{d_eqcomp},
                        stream.value());
    };

    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;

    if (compare_nans == nan_equality::ALL_EQUAL) {
      check_contains(nan_equal_comparator{});
    } else {
      check_contains(nan_unequal_comparator{});
    }
  }

  return contained;
}

}  // namespace cudf::detail
