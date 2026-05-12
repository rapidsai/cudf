/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace CUDF_EXPORT cudf {
namespace tags {

template <typename T = void>
inline constexpr char const* tag_of = "void";

#define CUDF_TYPE_TAG(type, tag) \
  using tag = type;              \
  template <>                    \
  inline constexpr char const* tag_of<type> = #tag;

CUDF_TYPE_TAG(bool, bool8);
CUDF_TYPE_TAG(int8_t, int8);
CUDF_TYPE_TAG(int16_t, int16);
CUDF_TYPE_TAG(int32_t, int32);
CUDF_TYPE_TAG(int64_t, int64);
CUDF_TYPE_TAG(__int128_t, int128);
CUDF_TYPE_TAG(uint8_t, uint8);
CUDF_TYPE_TAG(uint16_t, uint16);
CUDF_TYPE_TAG(uint32_t, uint32);
CUDF_TYPE_TAG(uint64_t, uint64);
CUDF_TYPE_TAG(__uint128_t, uint128);
CUDF_TYPE_TAG(float, float32);
CUDF_TYPE_TAG(double, float64);
CUDF_TYPE_TAG(cudf::string_view, string_view);
CUDF_TYPE_TAG(numeric::decimal32, decimal32);
CUDF_TYPE_TAG(numeric::decimal64, decimal64);
CUDF_TYPE_TAG(numeric::decimal128, decimal128);
CUDF_TYPE_TAG(cudf::duration_D, duration_D);
CUDF_TYPE_TAG(cudf::duration_h, duration_h);
CUDF_TYPE_TAG(cudf::duration_m, duration_m);
CUDF_TYPE_TAG(cudf::duration_s, duration_s);
CUDF_TYPE_TAG(cudf::duration_ms, duration_ms);
CUDF_TYPE_TAG(cudf::duration_us, duration_us);
CUDF_TYPE_TAG(cudf::duration_ns, duration_ns);
CUDF_TYPE_TAG(cudf::timestamp_D, timestamp_D);
CUDF_TYPE_TAG(cudf::timestamp_h, timestamp_h);
CUDF_TYPE_TAG(cudf::timestamp_m, timestamp_m);
CUDF_TYPE_TAG(cudf::timestamp_s, timestamp_s);
CUDF_TYPE_TAG(cudf::timestamp_ms, timestamp_ms);
CUDF_TYPE_TAG(cudf::timestamp_us, timestamp_us);
CUDF_TYPE_TAG(cudf::timestamp_ns, timestamp_ns);

}  // namespace tags
}  // namespace CUDF_EXPORT cudf
