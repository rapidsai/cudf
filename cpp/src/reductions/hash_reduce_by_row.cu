/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "hash_reduce_by_row.cuh"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

#if 0
rmm::device_uvector<size_type> hash_reduce_by_row(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
  size_type num_rows,
  cudf::nullate::DYNAMIC has_nulls,
  bool has_nested_columns,
  duplicate_keep_option keep,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(keep != duplicate_keep_option::KEEP_ANY,
               "This function should not be called with KEEP_ANY");

  auto reduction_results = rmm::device_uvector<size_type>(num_rows, stream, mr);

  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             reduction_results.begin(),
                             reduction_results.end(),
                             reduction_init_value(keep));

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = experimental::compaction_hash(row_hasher.device_hasher(has_nulls));

  auto const row_comp = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const reduce_by_row = [&](auto const value_comp) {
    if (has_nested_columns) {
      auto const key_equal = row_comp.equal_to<true>(has_nulls, nulls_equal, value_comp);
      thrust::for_each(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        reduce_by_row_fn{
          map.get_device_view(), key_hasher, key_equal, keep, reduction_results.begin()});
    } else {
      auto const key_equal = row_comp.equal_to<false>(has_nulls, nulls_equal, value_comp);
      thrust::for_each(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        reduce_by_row_fn{
          map.get_device_view(), key_hasher, key_equal, keep, reduction_results.begin()});
    }
  };

  if (nans_equal == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    reduce_by_row(nan_equal_comparator{});
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    reduce_by_row(nan_unequal_comparator{});
  }

  return reduction_results;
}
#endif

}  // namespace cudf::detail
