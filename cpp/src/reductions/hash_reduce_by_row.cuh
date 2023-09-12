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

#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf::detail {

/**
 * @brief Perform a reduction on groups of rows that are compared equal.
 *
 * This is essentially a reduce-by-key operation with keys are non-contiguous rows and are compared
 * equal. A hash table is used to find groups of equal rows.
 *
 * Depending on the `keep` parameter, the reduction operation for each row group is:
 * - If `keep == KEEP_FIRST`: min of row indices in the group.
 * - If `keep == KEEP_LAST`: max of row indices in the group.
 * - If `keep == KEEP_NONE`: count of equivalent rows (group size).
 *
 * At the beginning of the operation, the entire output array is filled with a value given by
 * the `reduction_init_value()` function. Then, the reduction result for each row group is written
 * into the output array at the index of an unspecified row in the group.
 *
 * @param map The auxiliary map to perform reduction
 * @param preprocessed_input The preprocessed of the input rows for computing row hashing and row
 *        comparisons
 * @param num_rows The number of all input rows
 * @param has_nulls Indicate whether the input rows has any nulls at any nested levels
 * @param has_nested_columns Indicates whether the input table has any nested columns
 * @param keep The parameter to determine what type of reduction to perform
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device_uvector containing the reduction results
 */
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
  rmm::mr::device_memory_resource* mr);

/**
 * @brief A functor to perform reduce-by-key with keys are rows that compared equal.
 *
 * TODO: We need to switch to use `static_reduction_map` when it is ready
 * (https://github.com/NVIDIA/cuCollections/pull/98).
 */
template <typename MapView, typename KeyHasher, typename KeyEqual, typename OutputType>
struct reduce_by_row_fn_base {
  MapView const d_map;
  KeyHasher const d_hasher;
  KeyEqual const d_equal;
  OutputType* const d_output;

  reduce_by_row_fn_base(MapView const& d_map,
                        KeyHasher const& d_hasher,
                        KeyEqual const& d_equal,
                        OutputType* const d_output)
    : d_map{d_map}, d_hasher{d_hasher}, d_equal{d_equal}, d_output{d_output}
  {
  }

 protected:
  __device__ OutputType* get_output_ptr(size_type const idx) const
  {
    auto const iter = d_map.find(idx, d_hasher, d_equal);

    if (iter != d_map.end()) {
      // Only one index value of the duplicate rows could be inserted into the map.
      // As such, looking up for all indices of duplicate rows always returns the same value.
      auto const inserted_idx = iter->second.load(cuda::std::memory_order_relaxed);

      // All duplicate rows will have concurrent access to this same output slot.
      return &d_output[inserted_idx];
    } else {
      // All input `idx` values have been inserted into the map before.
      // Thus, searching for an `idx` key resulting in the `end()` iterator only happens if
      // `d_equal(idx, idx) == false`.
      // Such situations are due to comparing nulls or NaNs which are considered as always unequal.
      // In those cases, all rows containing nulls or NaNs are distinct. Just return their direct
      // output slot.
      return &d_output[idx];
    }
  }
};

template <typename ReduceFuncBuilder, typename OutputType>
rmm::device_uvector<size_type> hash_reduce_by_row(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
  size_type num_rows,
  cudf::nullate::DYNAMIC has_nulls,
  bool has_nested_columns,
  null_equality nulls_equal,
  nan_equality nans_equal,
  ReduceFuncBuilder func_builder,
  OutputType init,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto reduction_results = rmm::device_uvector<OutputType>(num_rows, stream, mr);

  thrust::uninitialized_fill(
    rmm::exec_policy(stream), reduction_results.begin(), reduction_results.end(), init);

  auto const map_dview  = map.get_device_view();
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
        func_builder.build(map_dview, key_hasher, key_equal, reduction_results.begin()));
    } else {
      auto const key_equal = row_comp.equal_to<false>(has_nulls, nulls_equal, value_comp);
      thrust::for_each(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        func_builder.build(map_dview, key_hasher, key_equal, reduction_results.begin()));
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

}  // namespace cudf::detail
