/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/stream_compaction.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/hyperloglog.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>

namespace cudf {
namespace detail {

approx_distinct_count::~approx_distinct_count() = default;

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             cudf::size_type precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl{cuco::sketch_size_kb{static_cast<double>(
            4 * (1ull << std::max(cudf::size_type{4}, std::min(cudf::size_type{18}, precision))) /
            1024.0)},
          cuda::std::identity{},
          rmm::mr::polymorphic_allocator<cuda::std::byte>{},
          cuda::stream_ref{stream.value()}}
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return; }

  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key   = row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(has_nulls);

  auto const iter = thrust::counting_iterator<cudf::size_type>(0);

  rmm::device_uvector<uint64_t> hash_values(num_rows, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream), iter, iter + num_rows, hash_values.begin(), hash_key);

  _impl.add(hash_values.begin(), hash_values.end(), cuda::stream_ref{stream.value()});
}

void approx_distinct_count::add(table_view const& input,
                                null_policy null_handling,
                                nan_policy nan_handling,
                                rmm::cuda_stream_view stream)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return; }

  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key   = row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(has_nulls);

  auto const iter = thrust::counting_iterator<cudf::size_type>(0);

  rmm::device_uvector<uint64_t> hash_values(num_rows, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream), iter, iter + num_rows, hash_values.begin(), hash_key);

  _impl.add(hash_values.begin(), hash_values.end(), cuda::stream_ref{stream.value()});
}

cudf::size_type approx_distinct_count::estimate(rmm::cuda_stream_view stream) const
{
  return static_cast<cudf::size_type>(_impl.estimate(cuda::stream_ref{stream.value()}));
}

}  // namespace detail

// Public API implementations
approx_distinct_count::~approx_distinct_count() = default;

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             cudf::size_type precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl(std::make_unique<impl_type>(input, precision, null_handling, nan_handling, stream))
{
}

void approx_distinct_count::add(table_view const& input,
                                null_policy null_handling,
                                nan_policy nan_handling,
                                rmm::cuda_stream_view stream)
{
  _impl->add(input, null_handling, nan_handling, stream);
}

cudf::size_type approx_distinct_count::estimate(rmm::cuda_stream_view stream) const
{
  return _impl->estimate(stream);
}

}  // namespace cudf
