/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
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

namespace cudf {

namespace {

template <typename T>
struct valid_range {
  T start, stop;
  __device__ constexpr bool operator()(T index) const { return index >= start && index < stop; }
};

}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join_complement(cudf::device_span<size_type const> right_indices,
                                size_type probe_table_num_rows,
                                size_type build_table_num_rows,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto right_indices_complement =
    std::make_unique<rmm::device_uvector<size_type>>(build_table_num_rows, stream, mr);

  if (probe_table_num_rows == 0) {
    // All build rows are unmatched
    thrust::sequence(rmm::exec_policy_nosync(stream),
                     right_indices_complement->begin(),
                     right_indices_complement->end(),
                     0);
  } else {
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(build_table_num_rows, stream);
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                               invalid_index_map->begin(),
                               invalid_index_map->end(),
                               int32_t{1});

    valid_range<size_type> valid{0, build_table_num_rows};

    thrust::scatter_if(rmm::exec_policy_nosync(stream),
                       cuda::make_constant_iterator(0),
                       cuda::make_constant_iterator(0) + right_indices.size(),
                       right_indices.begin(),
                       right_indices.begin(),
                       invalid_index_map->begin(),
                       valid);

    auto const begin_counter = static_cast<size_type>(0);
    auto const end_counter   = static_cast<size_type>(build_table_num_rows);

    size_type const indices_count = thrust::copy_if(rmm::exec_policy_nosync(stream),
                                                    cuda::counting_iterator{begin_counter},
                                                    cuda::counting_iterator{end_counter},
                                                    invalid_index_map->begin(),
                                                    right_indices_complement->begin(),
                                                    cuda::std::identity{}) -
                                    right_indices_complement->begin();
    right_indices_complement->resize(indices_count, stream);
  }

  auto left_invalid_indices =
    std::make_unique<rmm::device_uvector<size_type>>(right_indices_complement->size(), stream, mr);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             left_invalid_indices->begin(),
                             left_invalid_indices->end(),
                             cudf::JoinNoMatch);

  return std::pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

}  // namespace cudf
