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

#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

bool contains_nested_element(column_view const& haystack,
                             column_view const& needle,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(needle.size() > 0, "Input needle column should have at least ONE row.");

  auto const haystack_tv = table_view{{haystack}};
  auto const needle_tv   = table_view{{needle}};
  auto const has_nulls   = has_nested_nulls(haystack_tv) || has_nested_nulls(needle_tv);

  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator(haystack_tv, needle_tv, stream);
  auto const d_comp = comparator.device_comparator(nullate::DYNAMIC{has_nulls});

  auto const begin = cudf::experimental::row::lhs_iterator(0);
  auto const end   = begin + haystack.size();
  using cudf::experimental::row::rhs_index_type;

  if (haystack.has_nulls()) {
    auto const haystack_cdv_ptr  = column_device_view::create(haystack, stream);
    auto const haystack_valid_it = cudf::detail::make_validity_iterator<false>(*haystack_cdv_ptr);

    return thrust::any_of(
      rmm::exec_policy(stream), begin, end, [d_comp, haystack_valid_it] __device__(auto const idx) {
        if (!haystack_valid_it[static_cast<size_type>(idx)]) { return false; }
        return d_comp(idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
      });
  }

  return thrust::any_of(rmm::exec_policy(stream), begin, end, [d_comp] __device__(auto const idx) {
    return d_comp(idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
  });
}

std::unique_ptr<column> multi_contains_nested_elements(column_view const& haystack,
                                                       column_view const& needles,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  auto result = make_numeric_column(data_type{type_to_id<bool>()},
                                    needles.size(),
                                    copy_bitmask(needles),
                                    needles.null_count(),
                                    stream,
                                    mr);
  if (needles.is_empty()) { return result; }

  auto const out_begin = result->mutable_view().template begin<bool>();
  if (haystack.is_empty()) {
    thrust::uninitialized_fill(
      rmm::exec_policy(stream), out_begin, out_begin + needles.size(), false);
    return result;
  }

  auto const haystack_tv        = table_view{{haystack}};
  auto const needles_tv         = table_view{{needles}};
  auto const haystack_has_nulls = has_nested_nulls(haystack_tv);
  auto const needles_has_nulls  = has_nested_nulls(needles_tv);

  auto haystack_map =
    detail::hash_map_type{compute_hash_table_size(haystack.size()),
                          detail::COMPACTION_EMPTY_KEY_SENTINEL,
                          detail::COMPACTION_EMPTY_VALUE_SENTINEL,
                          detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                          stream.value()};

  // Insert all indices of the elements in the haystack column into the hash map.
  // As such, we will use `thrust::equal_to` as key comparator.
  //
  // An alternative way to this is to use `self_comparator` for key comparisons, which would only
  // insert unique rows of the haystack column. This would save some memory (or significant amount
  // of memory if the haystack column contains many duplicate rows), however, will result in much
  // more processing time due to expensive row comparisons.
  {
    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      0, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

    auto const hasher   = cudf::experimental::row::hash::row_hasher(haystack_tv, stream);
    auto const d_hasher = detail::experimental::compaction_hash(
      hasher.device_hasher(nullate::DYNAMIC{haystack_has_nulls}));

    haystack_map.insert(haystack_it,
                        haystack_it + haystack.size(),
                        d_hasher,
                        thrust::equal_to<size_type>{},
                        stream.value());
  }

  // Check for existence of needles in haystack.
  // During this, we will use `table_comparator_adapter` to convert the existing indices in the
  // hash map (i.e., indices of haystack elements) into `lhs_index_type`, and indices of the
  // searching needles into `rhs_index_type` for row comparisons.
  {
    // Supply negative indices so `table_comparator_adapter` can recognize and convert them to
    // `rhs_index_type`.
    // A reverse iterator constructed from `0` value will begin from `-1`.
    // Thus, needle indices will iterate in reverse order in the range `[-1, -1-neeedles.size())`.
    // They need to be converted back to the range `[0, needles.size())` when calling to table
    // comparator and row hasher.
    auto const needles_it = thrust::make_reverse_iterator(thrust::make_counting_iterator(0));

    auto const hasher   = cudf::experimental::row::hash::row_hasher(needles_tv, stream);
    auto const d_hasher = cudf::experimental::row::hash::index_normalized_hasher_adapter{
      detail::experimental::compaction_hash(
        hasher.device_hasher(nullate::DYNAMIC{needles_has_nulls}))};

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(haystack_tv, needles_tv, stream);
    auto const d_comp = cudf::experimental::row::equality::index_normalized_comparator_adapter{
      comparator.device_comparator(nullate::DYNAMIC{haystack_has_nulls || needles_has_nulls})};

    haystack_map.contains(
      needles_it, needles_it + needles.size(), out_begin, d_hasher, d_comp, stream.value());
  }

  return result;
}

}  // namespace cudf::detail
