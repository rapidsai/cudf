/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/device/device_for.cuh>

namespace cudf {
namespace hashing {
namespace detail {

template <template <template <typename> class, typename> class DeviceRowHasher>
void hash_rows(cudf::detail::row::hash::row_hasher const& row_hasher,
               size_type num_rows,
               hash_value_type* output_begin,
               bool nullable,
               uint32_t seed,
               rmm::cuda_stream_view stream)
{
  auto const hasher = row_hasher.device_hasher<MurmurHash3_x86_32, DeviceRowHasher>(nullable, seed);
  // thrust::tabulate is slow here, see NVIDIA/cccl#9070
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    num_rows,
    [output_begin, hasher] __device__(size_type i) mutable { output_begin[i] = hasher(i); },
    stream.value()));
}

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  if (input.num_rows() == 0) { return output; }

  bool const nullable     = has_nulls(input);
  auto const row_hasher   = cudf::detail::row::hash::row_hasher(input, stream);
  auto const output_begin = output->mutable_view().begin<hash_value_type>();

  if (input.num_columns() > 1) {
    hash_rows<cudf::detail::row::hash::device_row_hasher_iterative>(
      row_hasher, input.num_rows(), output_begin, nullable, seed, stream);
  } else {
    hash_rows<cudf::detail::row::hash::device_row_hasher>(
      row_hasher, input.num_rows(), output_begin, nullable, seed, stream);
  }

  return output;
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
