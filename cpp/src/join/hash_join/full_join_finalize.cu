/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

#include <vector>

namespace cudf {
namespace {

template <typename T>
struct valid_range {
  T start, stop;
  __device__ constexpr bool operator()(T index) const { return index >= start && index < stop; }
};

/**
 * @brief Writes the unmatched build-row indices into `comp_right_out` and returns the count.
 */
size_type compute_complement(cudf::device_span<size_type const> right_indices,
                             size_type probe_table_num_rows,
                             size_type build_table_num_rows,
                             size_type* comp_right_out,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  if (probe_table_num_rows == 0) {
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     comp_right_out,
                     comp_right_out + build_table_num_rows,
                     0);
    return build_table_num_rows;
  }

  rmm::device_uvector<size_type> invalid_index_map(build_table_num_rows, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    invalid_index_map.begin(),
    invalid_index_map.end(),
    size_type{1});

  thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::make_constant_iterator(size_type{0}),
                     cuda::make_constant_iterator(size_type{0}) + right_indices.size(),
                     right_indices.begin(),
                     right_indices.begin(),
                     invalid_index_map.begin(),
                     valid_range<size_type>{0, build_table_num_rows});

  auto const end =
    thrust::copy_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{build_table_num_rows},
                    invalid_index_map.begin(),
                    comp_right_out,
                    cuda::std::identity{});
  return static_cast<size_type>(end - comp_right_out);
}

}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join_finalize(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type probe_table_num_rows,
  size_type build_table_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(left_partials.size() == right_partials.size(),
               "left_partials and right_partials must have the same length",
               std::invalid_argument);

  // Sum the probe partial sizes.
  std::size_t probe_total = 0;
  for (auto const& span : left_partials) {
    probe_total += span.size();
  }

  // Upper-bound the output at (probe matches + all build rows unmatched).
  auto const upper_bound = probe_total + static_cast<std::size_t>(build_table_num_rows);
  auto left_out  = std::make_unique<rmm::device_uvector<size_type>>(upper_bound, stream, mr);
  auto right_out = std::make_unique<rmm::device_uvector<size_type>>(upper_bound, stream, mr);

  // Concatenate every partial's (left, right) indices into the head of the output with a
  // single batched memcpy (one driver submission regardless of partition count).
  if (probe_total > 0) {
    auto const n = left_partials.size();
    std::vector<void*> dsts;
    std::vector<void const*> srcs;
    std::vector<std::size_t> sizes;
    dsts.reserve(2 * n);
    srcs.reserve(2 * n);
    sizes.reserve(2 * n);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n; ++i) {
      CUDF_EXPECTS(left_partials[i].size() == right_partials[i].size(),
                   "matching partials must have equal left/right sizes",
                   std::invalid_argument);
      auto const sz = left_partials[i].size() * sizeof(size_type);
      dsts.push_back(left_out->data() + offset);
      srcs.push_back(left_partials[i].data());
      sizes.push_back(sz);
      dsts.push_back(right_out->data() + offset);
      srcs.push_back(right_partials[i].data());
      sizes.push_back(sz);
      offset += left_partials[i].size();
    }
    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      dsts.data(), srcs.data(), sizes.data(), dsts.size(), stream));
  }

  // Append the complement into the tail of the output buffers.
  auto const comp_size =
    compute_complement(cudf::device_span<size_type const>{right_out->data(), probe_total},
                       probe_table_num_rows,
                       build_table_num_rows,
                       right_out->data() + probe_total,
                       stream,
                       mr);

  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    left_out->begin() + probe_total,
    left_out->begin() + probe_total + comp_size,
    cudf::JoinNoMatch);

  auto const final_size = probe_total + static_cast<std::size_t>(comp_size);
  left_out->resize(final_size, stream);
  right_out->resize(final_size, stream);

  return std::pair(std::move(left_out), std::move(right_out));
}

}  // namespace cudf
