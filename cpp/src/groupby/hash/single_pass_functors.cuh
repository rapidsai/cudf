/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "helpers.cuh"

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/utilities/assert.cuh>

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

/**
 * @brief Base struct to compute single-pass aggregations and store the results into an output
 * table, executing for all input rows.
 */
struct compute_single_pass_aggs_base_fn {
  aggregation::Kind const* aggs;
  table_device_view input_values;
  mutable_table_device_view output_values;

  compute_single_pass_aggs_base_fn(aggregation::Kind const* aggs,
                                   table_device_view const& input_values,
                                   mutable_table_device_view const& output_values)
    : aggs(aggs), input_values(input_values), output_values(output_values)
  {
  }
};

/**
 * @brief Functor to compute single-pass aggregations and store the results into an output table,
 * executing for all input rows.
 *
 * This functor writes output to the sparse intermediate output table, using the target indices
 * computed on-the-fly. In addition, aggregations are computed in serial order for each row.
 *
 * @tparam SetType Type of the key hash set
 */
template <typename SetRef>
struct compute_single_pass_aggs_sparse_output_fn : compute_single_pass_aggs_base_fn {
  SetRef set_ref;
  bitmask_type const* row_bitmask;

  compute_single_pass_aggs_sparse_output_fn(SetRef set_ref,
                                            bitmask_type const* row_bitmask,
                                            aggregation::Kind const* aggs,
                                            table_device_view const& input_values,
                                            mutable_table_device_view const& output_values)
    : compute_single_pass_aggs_base_fn(aggs, input_values, output_values),
      set_ref{set_ref},
      row_bitmask{row_bitmask}
  {
  }

  __device__ void operator()(size_type idx)
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, idx)) { return; }
    auto const target_row_idx = *set_ref.insert_and_find(idx).first;

    for (size_type col_idx = 0; col_idx < input_values.num_columns(); ++col_idx) {
      auto const& source_col = input_values.column(col_idx);
      auto const& target_col = output_values.column(col_idx);
      dispatch_type_and_aggregation(source_col.type(),
                                    aggs[col_idx],
                                    cudf::detail::element_aggregator{},
                                    target_col,
                                    target_row_idx,
                                    source_col,
                                    idx);
    }
  }
};

/**
 * @brief Functor to compute single-pass aggregations and store the results into an output table,
 * executing for all input rows.
 *
 * This functor writes output to the final dense output table, using the given pre-computed target
 * indices. In addition, all aggregations for all rows are computed concurrently without any order.
 */
struct compute_single_pass_aggs_dense_output_fn : compute_single_pass_aggs_base_fn {
  size_type const* target_indices;

  compute_single_pass_aggs_dense_output_fn(size_type const* target_indices,
                                           aggregation::Kind const* aggs,
                                           table_device_view const& input_values,
                                           mutable_table_device_view const& output_values)
    : compute_single_pass_aggs_base_fn(aggs, input_values, output_values),
      target_indices(target_indices)
  {
  }

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = input_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    if (auto const target_row_idx = target_indices[source_row_idx];
        target_row_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      auto const col_idx     = static_cast<size_type>(idx / num_rows);
      auto const& source_col = input_values.column(col_idx);
      auto const& target_col = output_values.column(col_idx);
      dispatch_type_and_aggregation(source_col.type(),
                                    aggs[col_idx],
                                    cudf::detail::element_aggregator{},
                                    target_col,
                                    target_row_idx,
                                    source_col,
                                    source_row_idx);
    }
  }
};

}  // namespace cudf::groupby::detail::hash
