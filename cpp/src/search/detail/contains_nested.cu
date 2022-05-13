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

namespace cudf {
namespace detail {

/**
 * @brief Check if the (unique) row of the @p value column is contained in the @p col column.
 *
 * This utility function is only applied for nested types (struct + list). Caller is responsible
 * to make sure the @p value column has EXACTLY ONE ROW.
 */
auto check_contain_scalar(column_view const& col,
                          column_view const& value,
                          rmm::cuda_stream_view stream)
{
  auto const col_tview     = table_view{{col}};
  auto const val_tview     = table_view{{value}};
  auto const has_any_nulls = has_nested_nulls(col_tview) || has_nested_nulls(val_tview);

  auto const comp =
    cudf::experimental::row::equality::table_comparator(col_tview, val_tview, stream);
  auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

  auto const col_cdv_ptr       = column_device_view::create(col, stream);
  auto const col_validity_iter = cudf::detail::make_validity_iterator<true>(*col_cdv_ptr);
  auto const begin             = thrust::make_counting_iterator(0);
  auto const end               = begin + col.size();
  auto const found_it          = thrust::find_if(
    rmm::exec_policy(stream), begin, end, [dcomp, col_validity_iter] __device__(auto const idx) {
      if (!col_validity_iter[idx]) { return false; }
      return dcomp(idx, 0);  // compare col[idx] == val[0].
    });

  return found_it != end;
}

auto check_contain_column(column_view const& values /* => haystack */,
                          column_view const& input /* => needles */,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
{
  auto const input_size  = input.size();
  auto const values_size = values.size();
  auto result            = make_numeric_column(data_type{type_to_id<bool>()},
                                    values_size,
                                    copy_bitmask(values),
                                    values.null_count(),
                                    stream,
                                    mr);
  if (values.is_empty()) { return result; }

  auto const out_begin = result->mutable_view().template begin<bool>();
  if (input.is_empty()) {
    thrust::uninitialized_fill(rmm::exec_policy(stream), out_begin, out_begin + values_size, false);
    return result;
  }

  auto const input_tview   = table_view{{input}};
  auto const val_tview     = table_view{{values}};
  auto const has_any_nulls = has_nested_nulls(input_tview) || has_nested_nulls(val_tview);

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input_tview, stream);
  auto input_map =
    detail::hash_map_type{compute_hash_table_size(input_size),
                          detail::COMPACTION_EMPTY_KEY_SENTINEL,
                          detail::COMPACTION_EMPTY_VALUE_SENTINEL,
                          detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                          stream.value()};

  auto const row_hash = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const hash_input =
    detail::experimental::compaction_hash(row_hash.device_hasher(has_any_nulls));

  auto const comp =
    cudf::experimental::row::equality::table_comparator(input_tview, val_tview, stream);
  auto const dcomp = comp.device_comparator(nullate::DYNAMIC{has_any_nulls});

  // todo: make pair(i, i) type of left_index_type
  auto const pair_it = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });
  input_map.insert(pair_it, pair_it + input_size, hash_input, dcomp, stream.value());

  // todo: make count_it of type right_index_type
  auto const count_it = thrust::make_counting_iterator<size_type>(0);
  input_map.contains(count_it, count_it + values_size, out_begin, hash_input);

  return result;
}

}  // namespace detail

}  // namespace cudf
