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

#include <hash/unordered_multiset.cuh>
#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/search.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

/**
 * @brief Check if the (unique) row of the `needle` column is contained in the `haystack` column.
 *
 * If the input `needle` column has more than one row, only the first row will be considered.
 *
 * This function is designed for nested types. It can also work with non-nested types
 * but with lower performance due to the complexity of the implementation.
 */
bool contains_nested_element(column_view const& haystack,
                             column_view const& needle,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(needle.size() > 0, "Input needle column does not have any row.");

  auto const haystack_tv   = table_view{{haystack}};
  auto const needle_tv     = table_view{{needle}};
  auto const has_any_nulls = has_nested_nulls(haystack_tv) || has_nested_nulls(needle_tv);

  auto const comp =
    cudf::experimental::row::equality::table_comparator(haystack_tv, needle_tv, stream);
  auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

  // TODO: Make this strong typed index
  auto const begin    = thrust::make_counting_iterator(0);
  auto const end      = begin + haystack.size();
  auto const found_it = [&] {
    if (haystack.has_nulls()) {
      auto const haystack_cdv_ptr  = column_device_view::create(haystack, stream);
      auto const haystack_valid_it = cudf::detail::make_validity_iterator<false>(*haystack_cdv_ptr);
      return thrust::find_if(rmm::exec_policy(stream),
                             begin,
                             end,
                             [dcomp, haystack_valid_it] __device__(auto const idx) {
                               if (!haystack_valid_it[idx]) { return false; }
                               return dcomp(idx, 0);  // compare haystack[idx] == needle[0].
                             });

    } else {
      return thrust::find_if(
        rmm::exec_policy(stream), begin, end, [dcomp] __device__(auto const idx) {
          return dcomp(idx, 0);  // compare haystack[idx] == needle[0].
        });
    }
  }();

  return found_it != end;
}

/**
 * @brief Check if each row of the `needles` column is contained in the `haystack` column,
 * specialized for nested type.
 *
 * This function is designed for nested types. It can also work with non-nested types
 * but with lower performance due to the complexity of the implementation.
 *
 */
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

  auto const haystack_tv   = table_view{{haystack}};
  auto const needles_tv    = table_view{{needles}};
  auto const has_any_nulls = has_nested_nulls(haystack_tv) || has_nested_nulls(needles_tv);

  auto haystack_map =
    detail::hash_map_type{compute_hash_table_size(haystack.size()),
                          detail::COMPACTION_EMPTY_KEY_SENTINEL,
                          detail::COMPACTION_EMPTY_VALUE_SENTINEL,
                          detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                          stream.value()};

  // todo: create preprocess table for both row_hasher and table_comparator
  auto const row_hasher = cudf::experimental::row::hash::row_hasher(haystack_tv, stream);
  auto const haystack_hash =
    detail::experimental::compaction_hash(row_hasher.device_hasher(has_any_nulls));

  auto const comp =
    cudf::experimental::row::equality::table_comparator(haystack_tv, needles_tv, stream);
  auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

  // todo: make pair(i, i) type of left_index_type
  auto const pair_it = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });
  haystack_map.insert(pair_it, pair_it + haystack.size(), haystack_hash, dcomp, stream.value());

  // todo: make count_it of type right_index_type
  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  haystack_map.contains(count_it, count_it + needles.size(), out_begin, haystack_hash);

  return result;
}

}  // namespace cudf::detail
