/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "murmurhash3_x86_32_jit_hasher_decl.cuh"

#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::hashing::detail {

__device__ inline hash_value_type murmur_jit_hash_dispatcher(column_device_view col,
                                                             uint32_t seed,
                                                             bool const nullable,
                                                             size_type row_index)
{
  switch (col.type().id()) {
    case type_id::INT8:
      return murmur_jit_hasher<id_to_type<type_id::INT8>>(col, seed, nullable, row_index);
    case type_id::INT16:
      return murmur_jit_hasher<id_to_type<type_id::INT16>>(col, seed, nullable, row_index);
    case type_id::INT32:
      return murmur_jit_hasher<id_to_type<type_id::INT32>>(col, seed, nullable, row_index);
    case type_id::INT64:
      return murmur_jit_hasher<id_to_type<type_id::INT64>>(col, seed, nullable, row_index);
    case type_id::UINT8:
      return murmur_jit_hasher<id_to_type<type_id::UINT8>>(col, seed, nullable, row_index);
    case type_id::UINT16:
      return murmur_jit_hasher<id_to_type<type_id::UINT16>>(col, seed, nullable, row_index);
    case type_id::UINT32:
      return murmur_jit_hasher<id_to_type<type_id::UINT32>>(col, seed, nullable, row_index);
    case type_id::UINT64:
      return murmur_jit_hasher<id_to_type<type_id::UINT64>>(col, seed, nullable, row_index);
    case type_id::FLOAT32:
      return murmur_jit_hasher<id_to_type<type_id::FLOAT32>>(col, seed, nullable, row_index);
    case type_id::FLOAT64:
      return murmur_jit_hasher<id_to_type<type_id::FLOAT64>>(col, seed, nullable, row_index);
    case type_id::BOOL8:
      return murmur_jit_hasher<id_to_type<type_id::BOOL8>>(col, seed, nullable, row_index);
    case type_id::TIMESTAMP_DAYS:
      return murmur_jit_hasher<id_to_type<type_id::TIMESTAMP_DAYS>>(col, seed, nullable, row_index);
    case type_id::TIMESTAMP_SECONDS:
      return murmur_jit_hasher<id_to_type<type_id::TIMESTAMP_SECONDS>>(
        col, seed, nullable, row_index);
    case type_id::TIMESTAMP_MILLISECONDS:
      return murmur_jit_hasher<id_to_type<type_id::TIMESTAMP_MILLISECONDS>>(
        col, seed, nullable, row_index);
    case type_id::TIMESTAMP_MICROSECONDS:
      return murmur_jit_hasher<id_to_type<type_id::TIMESTAMP_MICROSECONDS>>(
        col, seed, nullable, row_index);
    case type_id::TIMESTAMP_NANOSECONDS:
      return murmur_jit_hasher<id_to_type<type_id::TIMESTAMP_NANOSECONDS>>(
        col, seed, nullable, row_index);
    case type_id::DURATION_DAYS:
      return murmur_jit_hasher<id_to_type<type_id::DURATION_DAYS>>(col, seed, nullable, row_index);
    case type_id::DURATION_SECONDS:
      return murmur_jit_hasher<id_to_type<type_id::DURATION_SECONDS>>(
        col, seed, nullable, row_index);
    case type_id::DURATION_MILLISECONDS:
      return murmur_jit_hasher<id_to_type<type_id::DURATION_MILLISECONDS>>(
        col, seed, nullable, row_index);
    case type_id::DURATION_MICROSECONDS:
      return murmur_jit_hasher<id_to_type<type_id::DURATION_MICROSECONDS>>(
        col, seed, nullable, row_index);
    case type_id::DURATION_NANOSECONDS:
      return murmur_jit_hasher<id_to_type<type_id::DURATION_NANOSECONDS>>(
        col, seed, nullable, row_index);
    case type_id::DICTIONARY32:
      return murmur_jit_hasher<id_to_type<type_id::DICTIONARY32>>(col, seed, nullable, row_index);
    case type_id::STRING:
      return murmur_jit_hasher<id_to_type<type_id::STRING>>(col, seed, nullable, row_index);
    case type_id::LIST:
      return murmur_jit_hasher<id_to_type<type_id::LIST>>(col, seed, nullable, row_index);
    case type_id::DECIMAL32:
      return murmur_jit_hasher<id_to_type<type_id::DECIMAL32>>(col, seed, nullable, row_index);
    case type_id::DECIMAL64:
      return murmur_jit_hasher<id_to_type<type_id::DECIMAL64>>(col, seed, nullable, row_index);
    case type_id::DECIMAL128:
      return murmur_jit_hasher<id_to_type<type_id::DECIMAL128>>(col, seed, nullable, row_index);
    case type_id::STRUCT:
      return murmur_jit_hasher<id_to_type<type_id::STRUCT>>(col, seed, nullable, row_index);
    default: CUDF_UNREACHABLE("Invalid type_id.");
  }
}

}  // namespace cudf::hashing::detail
