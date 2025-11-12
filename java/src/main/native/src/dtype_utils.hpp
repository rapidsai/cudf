/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <jni.h>

namespace cudf {
namespace jni {

// convert a timestamp type to the corresponding duration type
inline cudf::data_type timestamp_to_duration(cudf::data_type dt)
{
  cudf::type_id duration_type_id;
  switch (dt.id()) {
    case cudf::type_id::TIMESTAMP_DAYS: duration_type_id = cudf::type_id::DURATION_DAYS; break;
    case cudf::type_id::TIMESTAMP_SECONDS:
      duration_type_id = cudf::type_id::DURATION_SECONDS;
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      duration_type_id = cudf::type_id::DURATION_MILLISECONDS;
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      duration_type_id = cudf::type_id::DURATION_MICROSECONDS;
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      duration_type_id = cudf::type_id::DURATION_NANOSECONDS;
      break;
    default: throw std::logic_error("Unexpected type in timestamp_to_duration");
  }
  return cudf::data_type(duration_type_id);
}

inline bool is_decimal_type(cudf::type_id n_type)
{
  return n_type == cudf::type_id::DECIMAL32 || n_type == cudf::type_id::DECIMAL64 ||
         n_type == cudf::type_id::DECIMAL128;
}

// create data_type including scale for decimal type
inline cudf::data_type make_data_type(jint out_dtype, jint scale)
{
  cudf::type_id n_type = static_cast<cudf::type_id>(out_dtype);
  cudf::data_type n_data_type;
  if (is_decimal_type(n_type)) {
    n_data_type = cudf::data_type(n_type, scale);
  } else {
    n_data_type = cudf::data_type(n_type);
  }
  return n_data_type;
}

}  // namespace jni
}  // namespace cudf
