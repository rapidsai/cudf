/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#pragma once

#include "helpers.cuh"

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/utilities/traits.cuh>

#include <cuda/std/cstddef>

namespace cudf::groupby::detail::hash {
/// Functor used by type dispatcher returning the size of the underlying C++ type
struct size_of_functor {
  template <typename T>
  CUDF_HOST_DEVICE constexpr cudf::size_type operator()()
  {
    return sizeof(T);
  }
};

// TODO: TO BE REMOVED issue tracked via #17171
template <typename T, cudf::aggregation::Kind k>
__device__ constexpr bool is_supported()
{
  return cudf::is_fixed_width<T>() and
         ((k == cudf::aggregation::SUM) or (k == cudf::aggregation::SUM_OF_SQUARES) or
          (k == cudf::aggregation::MIN) or (k == cudf::aggregation::MAX) or
          (k == cudf::aggregation::COUNT_VALID) or (k == cudf::aggregation::COUNT_ALL) or
          (k == cudf::aggregation::ARGMIN) or (k == cudf::aggregation::ARGMAX) or
          (k == cudf::aggregation::STD) or (k == cudf::aggregation::VARIANCE) or
          (k == cudf::aggregation::PRODUCT) and cudf::detail::is_product_supported<T>());
}

template <typename T, cudf::aggregation::Kind k>
__device__ T identity_from_operator()
  requires(not std::is_same_v<cudf::detail::corresponding_operator_t<k>, void>)
{
  using DeviceType = cudf::device_storage_type_t<T>;
  return cudf::detail::corresponding_operator_t<k>::template identity<DeviceType>();
}

template <typename T, cudf::aggregation::Kind k, typename Enable = void>
__device__ T identity_from_operator()
  requires(std::is_same_v<cudf::detail::corresponding_operator_t<k>, void>)
{
  CUDF_UNREACHABLE("Unable to get identity/sentinel from device operator");
}

template <typename T, cudf::aggregation::Kind k>
__device__ T get_identity()
{
  if ((k == cudf::aggregation::ARGMAX) or (k == cudf::aggregation::ARGMIN)) {
    if constexpr (cudf::is_timestamp<T>()) {
      return k == cudf::aggregation::ARGMAX
               ? T{typename T::duration(cudf::detail::ARGMAX_SENTINEL)}
               : T{typename T::duration(cudf::detail::ARGMIN_SENTINEL)};
    } else {
      using DeviceType = cudf::device_storage_type_t<T>;
      return k == cudf::aggregation::ARGMAX
               ? static_cast<DeviceType>(cudf::detail::ARGMAX_SENTINEL)
               : static_cast<DeviceType>(cudf::detail::ARGMIN_SENTINEL);
    }
  }
  return identity_from_operator<T, k>();
}

template <typename Target, cudf::aggregation::Kind k, typename Enable = void>
struct initialize_target_element {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type idx) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element<Target, k, std::enable_if_t<is_supported<Target, k>()>> {
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type idx) const noexcept
  {
    using DeviceType          = cudf::device_storage_type_t<Target>;
    DeviceType* target_casted = reinterpret_cast<DeviceType*>(target);

    target_casted[idx] = get_identity<DeviceType, k>();

    target_mask[idx] = (k == cudf::aggregation::COUNT_ALL) or (k == cudf::aggregation::COUNT_VALID);
  }
};

struct initialize_shmem {
  template <typename Target, cudf::aggregation::Kind k>
  __device__ void operator()(cuda::std::byte* target,
                             bool* target_mask,
                             cudf::size_type idx) const noexcept
  {
    initialize_target_element<Target, k>{}(target, target_mask, idx);
  }
};

template <typename Target, cudf::aggregation::Kind k, typename Enable = void>
struct initialize_target_element_gmem {
  __device__ void operator()(cudf::mutable_column_device_view, cudf::size_type) const noexcept
  {
    CUDF_UNREACHABLE("Invalid source type and aggregation combination.");
  }
};

template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element_gmem<
  Target,
  k,
  std::enable_if_t<is_supported<Target, k>() && cudf::is_fixed_width<Target>() &&
                   !cudf::is_fixed_point<Target>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    using DeviceType                     = cudf::device_storage_type_t<Target>;
    target.element<Target>(target_index) = get_identity<DeviceType, k>();
  }
};

template <typename Target, cudf::aggregation::Kind k>
struct initialize_target_element_gmem<
  Target,
  k,
  std::enable_if_t<is_supported<Target, k>() && cudf::is_fixed_point<Target>()>> {
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    using DeviceType                         = cudf::device_storage_type_t<Target>;
    target.element<DeviceType>(target_index) = get_identity<DeviceType, k>();
  }
};

struct initialize_gmem {
  template <typename Target, cudf::aggregation::Kind k>
  __device__ void operator()(cudf::mutable_column_device_view target,
                             cudf::size_type target_index) const noexcept
  {
    initialize_target_element_gmem<Target, k>{}(target, target_index);
  }
};

struct initialize_sparse_table {
  cudf::size_type const* row_indices;
  cudf::mutable_table_device_view sparse_table;
  cudf::aggregation::Kind const* __restrict__ aggs;
  initialize_sparse_table(cudf::size_type const* row_indices,
                          cudf::mutable_table_device_view sparse_table,
                          cudf::aggregation::Kind const* aggs)
    : row_indices(row_indices), sparse_table(sparse_table), aggs(aggs)
  {
  }
  __device__ void operator()(cudf::size_type i)
  {
    auto key_idx = row_indices[i];
    for (auto col_idx = 0; col_idx < sparse_table.num_columns(); col_idx++) {
      cudf::detail::dispatch_type_and_aggregation(sparse_table.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  initialize_gmem{},
                                                  sparse_table.column(col_idx),
                                                  key_idx);
    }
  }
};

struct global_memory_fallback_fn {
  size_type const* key_indices;
  cudf::table_device_view input_values;
  cudf::mutable_table_device_view output_values;
  cudf::aggregation::Kind const* __restrict__ aggs;
  cudf::size_type const* fallback_block_ids;
  cudf::size_type stride;
  cudf::size_type num_strides;
  cudf::size_type full_stride;
  size_type num_processing_rows;
  size_type num_rows;

  global_memory_fallback_fn(size_type const* key_indices,
                            cudf::table_device_view input_values,
                            cudf::mutable_table_device_view output_values,
                            cudf::aggregation::Kind const* aggs,
                            cudf::size_type const* fallback_block_ids,
                            cudf::size_type stride,
                            cudf::size_type num_strides,
                            cudf::size_type full_stride,
                            size_type num_processing_rows,
                            size_type num_rows)
    : key_indices(key_indices),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      fallback_block_ids(fallback_block_ids),
      stride(stride),
      num_strides(num_strides),
      full_stride(full_stride),
      num_processing_rows(num_processing_rows),
      num_rows(num_rows)
  {
  }

  __device__ void operator()(int64_t idx) const
  {
    auto const agg_idx       = static_cast<size_type>(idx / num_processing_rows);
    auto const local_agg_idx = static_cast<size_type>(idx % num_processing_rows);
    auto const idx_in_agg    = local_agg_idx % stride;
    auto const thread_rank   = idx_in_agg % GROUPBY_BLOCK_SIZE;
    auto const block_idx     = fallback_block_ids[idx_in_agg / GROUPBY_BLOCK_SIZE];
    auto const row_idx =
      full_stride * (local_agg_idx / stride) + GROUPBY_BLOCK_SIZE * block_idx + thread_rank;
    if (row_idx >= num_rows) { return; }

    if (auto const target_idx = key_indices[row_idx];
        target_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      cudf::detail::aggregate_row(agg_idx, output_values, target_idx, input_values, row_idx, aggs);
    }
  }
};

/**
 * @brief Computes single-pass aggregations and store results into a sparse `output_values` table,
 * and populate `set` with indices of unique keys
 *
 * The hash set is built by inserting every row index `i` from the `keys` and `values` tables. If
 * the index was not present in the set, insert they index and then copy it to the output. If the
 * key was already present in the set, then the inserted index is aggregated with the existing row.
 * This aggregation is done for every element `j` in the row by applying aggregation operation `j`
 * between the new and existing element.
 *
 * Instead of storing the entire rows from `input_keys` and `input_values` in
 * the hashset, we instead store the row indices. For example, when inserting
 * row at index `i` from `input_keys` into the hash set, the value `i` is what
 * gets stored for the hash set's "key". It is assumed the `set` was constructed
 * with a custom comparator that uses these row indices to check for equality
 * between key rows. For example, comparing two keys `k0` and `k1` will compare
 * the two rows `input_keys[k0] ?= input_keys[k1]`
 *
 * The exact size of the result is not known a priori, but can be upper bounded
 * by the number of rows in `input_keys` & `input_values`. Therefore, it is
 * assumed `output_values` has sufficient storage for an equivalent number of
 * rows. In this way, after all rows are aggregated, `output_values` will likely
 * be "sparse", meaning that not all rows contain the result of an aggregation.
 *
 * @tparam SetType The type of the hash set device ref
 */
struct compute_single_pass_aggs_fn {
  size_type const* key_indices;
  table_device_view input_values;
  aggregation::Kind const* __restrict__ aggs;
  mutable_table_device_view output_values;

  /**
   * @brief Construct a new compute_single_pass_aggs_fn functor object
   *
   * @param set_ref Hash set object to insert key,value pairs into.
   * @param input_values The table whose rows will be aggregated in the values
   * of the hash set
   * @param aggs The set of aggregation operations to perform across the
   * @param output_values Table that stores the results of aggregating rows of
   * `input_values`.
   * columns of the `input_values` rows
   */
  compute_single_pass_aggs_fn(size_type const* key_indices,
                              table_device_view input_values,
                              aggregation::Kind const* aggs,
                              mutable_table_device_view output_values)
    : key_indices(key_indices), input_values(input_values), aggs(aggs), output_values(output_values)
  {
  }

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows = input_values.num_rows();
    auto const agg_idx  = static_cast<size_type>(idx / num_rows);
    auto const row_idx  = static_cast<size_type>(idx % num_rows);
    if (auto const target_idx = key_indices[row_idx];
        target_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      cudf::detail::aggregate_row(agg_idx, output_values, target_idx, input_values, row_idx, aggs);
    }
  }
};
}  // namespace cudf::groupby::detail::hash
