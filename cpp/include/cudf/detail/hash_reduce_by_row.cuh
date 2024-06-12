/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/static_map.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

using hash_map_type = cuco::legacy::
  static_map<size_type, size_type, cuda::thread_scope_device, cudf::detail::cuco_allocator>;

/**
 * @brief The base struct for customized reduction functor to perform reduce-by-key with keys are
 * rows that compared equal.
 *
 * TODO: We need to switch to use `static_reduction_map` when it is ready
 * (https://github.com/NVIDIA/cuCollections/pull/98).
 */
template <typename MapView, typename KeyHasher, typename KeyEqual, typename OutputType>
struct reduce_by_row_fn_base {
 protected:
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

  /**
   * @brief Return a pointer to the output array at the given index.
   *
   * @param idx The access index
   * @return A pointer to the given index in the output array
   */
  __device__ OutputType* get_output_ptr(size_type const idx) const
  {
    auto const iter = d_map.find(idx, d_hasher, d_equal);

    if (iter != d_map.end()) {
      // Only one (undetermined) index value of the duplicate rows could be inserted into the map.
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

/**
 * @brief Perform a reduction on groups of rows that are compared equal.
 *
 * This is essentially a reduce-by-key operation with keys are non-contiguous rows and are compared
 * equal. A hash table is used to find groups of equal rows.
 *
 * At the beginning of the operation, the entire output array is filled with a value given by
 * the `init` parameter. Then, the reduction result for each row group is written into the output
 * array at the index of an unspecified row in the group.
 *
 * @tparam ReduceFuncBuilder The builder class that must have a `build()` method returning a
 *         reduction functor derived from `reduce_by_row_fn_base`
 * @tparam OutputType Type of the reduction results
 * @param map The auxiliary map to perform reduction
 * @param preprocessed_input The preprocessed of the input rows for computing row hashing and row
 *        comparisons
 * @param num_rows The number of all input rows
 * @param has_nulls Indicate whether the input rows has any nulls at any nested levels
 * @param has_nested_columns Indicates whether the input table has any nested columns
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether NaN values in floating point column should be
 *        considered equal.
 * @param init The initial value for reduction of each row group
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device_uvector containing the reduction results
 */
template <typename ReduceFuncBuilder, typename OutputType>
rmm::device_uvector<OutputType> hash_reduce_by_row(
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
  rmm::device_async_resource_ref mr)
{
  auto const map_dview  = map.get_device_view();
  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = row_hasher.device_hasher(has_nulls);
  auto const row_comp   = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto reduction_results = rmm::device_uvector<OutputType>(num_rows, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), reduction_results.begin(), reduction_results.end(), init);

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
