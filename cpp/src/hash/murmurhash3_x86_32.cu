/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include "murmurhash3_x86_32_lto.cuh"

namespace cudf {
namespace hashing {
namespace detail {

//
// 1. Build a device-side dispatcher that 

//template <typename T>
//hash_value_type hasher(cudf::column_device_view col, uint32_t seed);

__device__  __forceinline__ constexpr decltype(auto) hash_dispatcher(cudf::column_device_view col, uint32_t seed, bool const nullable)
{
  switch (col.type().id()) {
    case type_id::INT8:
      return hasher<id_to_type<type_id::INT8>>(
        col, seed, nullable);
    case type_id::INT16:
      return hasher<id_to_type<type_id::INT16>>(
        col, seed, nullable);
    case type_id::INT32:
      return hasher<id_to_type<type_id::INT32>>(
        col, seed, nullable);
    case type_id::INT64:
      return hasher<id_to_type<type_id::INT64>>(
        col, seed, nullable);
    case type_id::UINT8:
      return hasher<id_to_type<type_id::UINT8>>(
        col, seed, nullable);
    case type_id::UINT16:
      return hasher<id_to_type<type_id::UINT16>>(
        col, seed, nullable);
    case type_id::UINT32:
      return hasher<id_to_type<type_id::UINT32>>(
        col, seed, nullable);
    case type_id::UINT64:
      return hasher<id_to_type<type_id::UINT64>>(
        col, seed, nullable);
    case type_id::FLOAT32:
      return hasher<id_to_type<type_id::FLOAT32>>(
        col, seed, nullable);
    case type_id::FLOAT64:
      return hasher<id_to_type<type_id::FLOAT64>>(
        col, seed, nullable);
    case type_id::BOOL8:
      return hasher<id_to_type<type_id::BOOL8>>(
        col, seed, nullable);
    case type_id::TIMESTAMP_DAYS:
      return hasher<id_to_type<type_id::TIMESTAMP_DAYS>>(
        col, seed, nullable);
    case type_id::TIMESTAMP_SECONDS:
      return hasher<id_to_type<type_id::TIMESTAMP_SECONDS>>(
        col, seed, nullable);
    case type_id::TIMESTAMP_MILLISECONDS:
      return hasher<id_to_type<type_id::TIMESTAMP_MILLISECONDS>>(
        col, seed, nullable);
    case type_id::TIMESTAMP_MICROSECONDS:
      return hasher<id_to_type<type_id::TIMESTAMP_MICROSECONDS>>(
        col, seed, nullable);
    case type_id::TIMESTAMP_NANOSECONDS:
      return hasher<id_to_type<type_id::TIMESTAMP_NANOSECONDS>>(
        col, seed, nullable);
    case type_id::DURATION_DAYS:
      return hasher<id_to_type<type_id::DURATION_DAYS>>(
        col, seed, nullable);
    case type_id::DURATION_SECONDS:
      return hasher<id_to_type<type_id::DURATION_SECONDS>>(
        col, seed, nullable);
    case type_id::DURATION_MILLISECONDS:
      return hasher<id_to_type<type_id::DURATION_MILLISECONDS>>(
        col, seed, nullable);
    case type_id::DURATION_MICROSECONDS:
      return hasher<id_to_type<type_id::DURATION_MICROSECONDS>>(
        col, seed, nullable);
    case type_id::DURATION_NANOSECONDS:
      return hasher<id_to_type<type_id::DURATION_NANOSECONDS>>(
        col, seed, nullable);
    case type_id::DICTIONARY32:
      return hasher<id_to_type<type_id::DICTIONARY32>>(
        col, seed, nullable);
    case type_id::STRING:
      return hasher<id_to_type<type_id::STRING>>(
        col, seed, nullable);
    case type_id::LIST:
      return hasher<id_to_type<type_id::LIST>>(
        col, seed, nullable);
    case type_id::DECIMAL32:
      return hasher<id_to_type<type_id::DECIMAL32>>(
        col, seed, nullable);
    case type_id::DECIMAL64:
      return hasher<id_to_type<type_id::DECIMAL64>>(
        col, seed, nullable);
    case type_id::DECIMAL128:
      return hasher<id_to_type<type_id::DECIMAL128>>(
        col, seed, nullable);
    case type_id::STRUCT:
      return hasher<id_to_type<type_id::STRUCT>>(
        col, seed, nullable);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid type_id.");
#else
      CUDF_UNREACHABLE("Invalid type_id.");
#endif
    }
  }
}

__global__ void murmurhash3_x86_32_kernel(mutable_column_device_view output,
                                  uint32_t seed,
                                  table_device_view const input,
                                  bool const nullable)
{
  cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < input.num_rows()) {
      auto const num_cols = input.num_columns();
      if (num_cols == 0) return;
      hash_value_type hash_value = hash_dispatcher(input.column(0), seed, nullable);
      for (int i = 1; i < num_cols; ++i) {
          hash_value = cudf::hashing::detail::hash_combine(
            hash_value, hash_dispatcher(input.column(i), seed, nullable));
      }
      output.element<hash_value_type>(idx) = hash_value;
  }
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

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nulls(input);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  //thrust::tabulate(rmm::exec_policy_nosync(stream),
  //                 output_view.begin<hash_value_type>(),
  //                 output_view.end<hash_value_type>(),
  //                 row_hasher.device_hasher<MurmurHash3_x86_32>(nullable, seed));
  //
  auto d_output = mutable_column_device_view::create(output_view, stream);
  auto d_input  = table_device_view::create(input, stream);
  murmurhash3_x86_32_kernel<<<1, input.num_rows(), 0, stream.value()>>>(
    *d_output, seed, *d_input, nullable);
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
