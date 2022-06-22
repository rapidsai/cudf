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

#include "distinct_reduce.cuh"

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

namespace {
/**
 * @brief A functor to perform reduce-by-key with keys are rows that compared equal.
 *
 * TODO: We need to switch to use `static_reduction_map` when it is ready
 * (https://github.com/NVIDIA/cuCollections/pull/98).
 */
template <typename MapView, typename KeyHasher, typename KeyEqual>
struct reduce_by_row_fn {
  MapView const d_map;
  KeyHasher const d_hasher;
  KeyEqual const d_equal;
  duplicate_keep_option const keep;
  size_type* const d_output;

  reduce_by_row_fn(MapView const& d_map,
                   KeyHasher const& d_hasher,
                   KeyEqual const& d_equal,
                   duplicate_keep_option const keep,
                   size_type* const d_output)
    : d_map{d_map}, d_hasher{d_hasher}, d_equal{d_equal}, keep{keep}, d_output{d_output}
  {
  }

  __device__ void operator()(size_type const idx) const
  {
    auto const out_ptr = get_output_ptr(idx);

    if (keep == duplicate_keep_option::KEEP_FIRST) {
      // Store the smallest index of all rows that are equal.
      atomicMin(out_ptr, idx);
    } else if (keep == duplicate_keep_option::KEEP_LAST) {
      // Store the greatest index of all rows that are equal.
      atomicMax(out_ptr, idx);
    } else {
      // Count the number of rows in each group of rows that are compared equal.
      atomicAdd(out_ptr, size_type{1});
    }
  }

 private:
  __device__ size_type* get_output_ptr(size_type const idx) const
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

}  // namespace

rmm::device_uvector<size_type> hash_reduce_by_row(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
  size_type num_rows,
  cudf::nullate::DYNAMIC has_nulls,
  duplicate_keep_option keep,
  null_equality nulls_equal,
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

  auto const row_comp  = cudf::experimental::row::equality::self_comparator(preprocessed_input);
  auto const key_equal = row_comp.equal_to(has_nulls, nulls_equal);

  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_rows),
    reduce_by_row_fn{
      map.get_device_view(), key_hasher, key_equal, keep, reduction_results.begin()});

  return reduction_results;
}

}  // namespace cudf::detail
