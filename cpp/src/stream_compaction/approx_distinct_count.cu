/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream_compaction_common.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/hyperloglog.cuh>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>

namespace cudf {
namespace detail {

// Internal implementation function
cudf::size_type approx_distinct_count_impl(table_view const& input,
                                           int precision,
                                           null_policy null_handling,
                                           nan_policy nan_handling,
                                           rmm::cuda_stream_view stream)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return 0; }

  // Clamp precision to valid range for HyperLogLog
  precision = std::max(4, std::min(18, precision));

  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key   = row_hasher.device_hasher(has_nulls);

  auto hll = cuco::hyperloglog<cudf::hash_value_type,
                               cuda::thread_scope_device,
                               cuco::xxhash_64<cudf::hash_value_type>,
                               rmm::mr::polymorphic_allocator<cuda::std::byte>>{
    cuco::sketch_size_kb{static_cast<double>(4 * (1ull << precision) / 1024.0)},
    cuco::xxhash_64<cudf::hash_value_type>{},
    rmm::mr::polymorphic_allocator<cuda::std::byte>{},
    cuda::stream_ref{stream.value()}};

  auto const iter = thrust::counting_iterator<cudf::size_type>(0);

  rmm::device_uvector<cudf::hash_value_type> hash_values(num_rows, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream), iter, iter + num_rows, hash_values.begin(), hash_key);

  // Create a temporary table for distinct processing if needed
  if (nan_handling == nan_policy::NAN_IS_NULL || null_handling == null_policy::EXCLUDE) {
    if (num_rows < 10000) {
      if (input.num_columns() == 1) {
        return cudf::distinct_count(input.column(0), null_handling, nan_handling);
      } else {
        return cudf::distinct_count(input, cudf::null_equality::EQUAL);
      }
    }
  }

  if (null_handling == null_policy::EXCLUDE && has_nulls) {
    auto const [row_bitmask, null_count] =
      cudf::detail::bitmask_or(input, stream, cudf::get_current_device_resource_ref());

    if (null_count > 0) {
      row_validity pred{static_cast<bitmask_type const*>(row_bitmask.data())};
      auto counting_iter = thrust::counting_iterator<size_type>(0);

      rmm::device_uvector<cudf::hash_value_type> filtered_hashes(num_rows - null_count, stream);
      auto end_iter = thrust::copy_if(rmm::exec_policy(stream),
                                      hash_values.begin(),
                                      hash_values.end(),
                                      counting_iter,
                                      filtered_hashes.begin(),
                                      pred);

      auto actual_count = std::distance(filtered_hashes.begin(), end_iter);
      if (actual_count > 0) {
        hll.add(filtered_hashes.begin(),
                filtered_hashes.begin() + actual_count,
                cuda::stream_ref{stream.value()});
      }
      return static_cast<cudf::size_type>(hll.estimate(cuda::stream_ref{stream.value()}));
    }
  }

  hll.add(hash_values.begin(), hash_values.end(), cuda::stream_ref{stream.value()});
  return static_cast<cudf::size_type>(hll.estimate(cuda::stream_ref{stream.value()}));
}

cudf::size_type approx_distinct_count(table_view const& input,
                                      int precision,
                                      null_policy null_handling,
                                      nan_policy nan_handling,
                                      rmm::cuda_stream_view stream)
{
  return approx_distinct_count_impl(input, precision, null_handling, nan_handling, stream);
}

cudf::size_type approx_distinct_count(column_view const& input,
                                      int precision,
                                      null_policy null_handling,
                                      nan_policy nan_handling,
                                      rmm::cuda_stream_view stream)
{
  // Convert column to single-column table and use unified implementation
  cudf::table_view single_col_table({input});
  return approx_distinct_count_impl(
    single_col_table, precision, null_handling, nan_handling, stream);
}

}  // namespace detail

// Public API implementations
cudf::size_type approx_distinct_count(column_view const& input,
                                      int precision,
                                      null_policy null_handling,
                                      nan_policy nan_handling,
                                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::approx_distinct_count(input, precision, null_handling, nan_handling, stream);
}

cudf::size_type approx_distinct_count(table_view const& input,
                                      int precision,
                                      null_policy null_handling,
                                      nan_policy nan_handling,
                                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::approx_distinct_count(input, precision, null_handling, nan_handling, stream);
}

}  // namespace cudf
