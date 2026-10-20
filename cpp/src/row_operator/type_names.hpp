/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <string>
namespace cudf::jit {
inline std::string row_abi_type(cudf::data_type type)
{
  switch (type.id()) {
    case type_id::BOOL8: return "bool8";
    case type_id::INT8: return "i8";
    case type_id::INT16: return "i16";
    case type_id::INT32: return "i32";
    case type_id::INT64: return "i64";
    case type_id::UINT8: return "u8";
    case type_id::UINT16: return "u16";
    case type_id::UINT32: return "u32";
    case type_id::UINT64: return "u64";
    case type_id::FLOAT32: return "f32";
    case type_id::FLOAT64: return "f64";
    case type_id::DECIMAL32: return "decimal32";
    case type_id::DECIMAL64: return "decimal64";
    case type_id::DECIMAL128: return "decimal128";
    case type_id::TIMESTAMP_DAYS: return "timestamp_days";
    case type_id::TIMESTAMP_SECONDS: return "timestamp_seconds";
    case type_id::TIMESTAMP_MILLISECONDS: return "timestamp_milliseconds";
    case type_id::TIMESTAMP_MICROSECONDS: return "timestamp_microseconds";
    case type_id::TIMESTAMP_NANOSECONDS: return "timestamp_nanoseconds";
    case type_id::DURATION_DAYS: return "duration_days";
    case type_id::DURATION_SECONDS: return "duration_seconds";
    case type_id::DURATION_MILLISECONDS: return "duration_milliseconds";
    case type_id::DURATION_MICROSECONDS: return "duration_microseconds";
    case type_id::DURATION_NANOSECONDS: return "duration_nanoseconds";
    case type_id::STRING: return "string_view";
    default: CUDF_FAIL("Unsupported row ABI element type", std::invalid_argument);
  }
}
}  // namespace cudf::jit
