/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/row_abi.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace cudf::row_abi::detail {
template <typename T>
struct codec;
template <>
struct codec<bool8> {
  using abi_type      = bool8;
  using physical_type = bool;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<i8> {
  using abi_type      = i8;
  using physical_type = int8_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<i16> {
  using abi_type      = i16;
  using physical_type = int16_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<i32> {
  using abi_type      = i32;
  using physical_type = int32_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<i64> {
  using abi_type      = i64;
  using physical_type = int64_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<u8> {
  using abi_type      = u8;
  using physical_type = uint8_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<u16> {
  using abi_type      = u16;
  using physical_type = uint16_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<u32> {
  using abi_type      = u32;
  using physical_type = uint32_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<u64> {
  using abi_type      = u64;
  using physical_type = uint64_t;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<f32> {
  using abi_type      = f32;
  using physical_type = float;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<f64> {
  using abi_type      = f64;
  using physical_type = double;
  __device__ static physical_type decode(abi_type v) { return static_cast<physical_type>(v); }
  __device__ static abi_type encode(physical_type v) { return static_cast<abi_type>(v); }
};
template <>
struct codec<decimal32> {
  using abi_type      = decimal32;
  using physical_type = numeric::decimal32;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{numeric::scaled_integer<typename physical_type::rep>{
      static_cast<typename physical_type::rep>(v.value), numeric::scale_type{v.scale}}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.value()), static_cast<i32>(v.scale())};
  }
};
template <>
struct codec<decimal64> {
  using abi_type      = decimal64;
  using physical_type = numeric::decimal64;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{numeric::scaled_integer<typename physical_type::rep>{
      static_cast<typename physical_type::rep>(v.value), numeric::scale_type{v.scale}}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.value()), static_cast<i32>(v.scale())};
  }
};
template <>
struct codec<decimal128> {
  using abi_type      = decimal128;
  using physical_type = numeric::decimal128;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{numeric::scaled_integer<typename physical_type::rep>{
      static_cast<typename physical_type::rep>(v.value), numeric::scale_type{v.scale}}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.value()), static_cast<i32>(v.scale())};
  }
};
template <>
struct codec<timestamp_days> {
  using abi_type      = timestamp_days;
  using physical_type = cudf::timestamp_D;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{typename physical_type::duration{v.value}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.time_since_epoch().count())};
  }
};
template <>
struct codec<timestamp_seconds> {
  using abi_type      = timestamp_seconds;
  using physical_type = cudf::timestamp_s;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{typename physical_type::duration{v.value}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.time_since_epoch().count())};
  }
};
template <>
struct codec<timestamp_milliseconds> {
  using abi_type      = timestamp_milliseconds;
  using physical_type = cudf::timestamp_ms;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{typename physical_type::duration{v.value}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.time_since_epoch().count())};
  }
};
template <>
struct codec<timestamp_microseconds> {
  using abi_type      = timestamp_microseconds;
  using physical_type = cudf::timestamp_us;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{typename physical_type::duration{v.value}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.time_since_epoch().count())};
  }
};
template <>
struct codec<timestamp_nanoseconds> {
  using abi_type      = timestamp_nanoseconds;
  using physical_type = cudf::timestamp_ns;
  __device__ static physical_type decode(abi_type v)
  {
    return physical_type{typename physical_type::duration{v.value}};
  }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.time_since_epoch().count())};
  }
};
template <>
struct codec<duration_days> {
  using abi_type      = duration_days;
  using physical_type = cudf::duration_D;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.value}; }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.count())};
  }
};
template <>
struct codec<duration_seconds> {
  using abi_type      = duration_seconds;
  using physical_type = cudf::duration_s;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.value}; }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.count())};
  }
};
template <>
struct codec<duration_milliseconds> {
  using abi_type      = duration_milliseconds;
  using physical_type = cudf::duration_ms;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.value}; }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.count())};
  }
};
template <>
struct codec<duration_microseconds> {
  using abi_type      = duration_microseconds;
  using physical_type = cudf::duration_us;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.value}; }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.count())};
  }
};
template <>
struct codec<duration_nanoseconds> {
  using abi_type      = duration_nanoseconds;
  using physical_type = cudf::duration_ns;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.value}; }
  __device__ static abi_type encode(physical_type v)
  {
    return {static_cast<decltype(abi_type::value)>(v.count())};
  }
};
template <>
struct codec<string_view> {
  using abi_type      = string_view;
  using physical_type = cudf::string_view;
  __device__ static physical_type decode(abi_type v) { return physical_type{v.data, v.bytes}; }
  __device__ static abi_type encode(physical_type v) { return {v.data(), v.size_bytes(), -1}; }
};
template <typename T>
__device__ auto decode(optional<T> v)
{
  using P = typename codec<T>::physical_type;
  return v.validity ? cuda::std::optional<P>{codec<T>::decode(v.value)} : cuda::std::optional<P>{};
}
template <typename T>
__device__ auto decode(T v)
{
  return codec<T>::decode(v);
}
}  // namespace cudf::row_abi::detail
