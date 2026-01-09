/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/jit/lto/types.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {

namespace lto {

template <typename T>
struct lowered_type_of_t;

template <typename T>
using lowered_type_of = typename lowered_type_of_t<T>::type;

template <typename T>
struct lifted_type_of_t;

template <typename T>
using lifted_type_of = typename lifted_type_of_t<T>::type;

#define CUDF_LTO_MAP(lowered_type, lifted_type)                                              \
  static_assert(sizeof(lowered_type) == sizeof(lifted_type),                                 \
                "(1: size). Lowered and Lifted types must be bitwise-equivalent");           \
  static_assert(alignof(lowered_type) == alignof(lifted_type),                               \
                "(2: alignment). Lowered and Lifted types must be bitwise-equivalent");      \
  static_assert(                                                                             \
    sizeof(lto::optional<lowered_type>) == sizeof(cuda::std::optional<lifted_type>),         \
    "(1: size). Lowered and Lifted types must have bitwise-equivalent optional types");      \
  static_assert(                                                                             \
    alignof(lto::optional<lowered_type>) == alignof(cuda::std::optional<lifted_type>),       \
    "(2: alignment). Lowered and Lifted types must have bitwise-equivalent optional types"); \
                                                                                             \
  template <>                                                                                \
  struct lifted_type_of_t<lowered_type> {                                                    \
    using type = lifted_type;                                                                \
  };                                                                                         \
                                                                                             \
  template <>                                                                                \
  struct lowered_type_of_t<lifted_type> {                                                    \
    using type = lowered_type;                                                               \
  };                                                                                         \
                                                                                             \
  __device__ __forceinline__ lowered_type* lower(lifted_type* p)                             \
  {                                                                                          \
    return reinterpret_cast<lowered_type*>(p);                                               \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ lowered_type const* lower(lifted_type const* p)                 \
  {                                                                                          \
    return reinterpret_cast<lowered_type const*>(p);                                         \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ lto::optional<lowered_type>* lower(                             \
    cuda::std::optional<lifted_type>* p)                                                     \
  {                                                                                          \
    return reinterpret_cast<lto::optional<lowered_type>*>(p);                                \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ lto::optional<lowered_type> const* lower(                       \
    cuda::std::optional<lifted_type> const* p)                                               \
  {                                                                                          \
    return reinterpret_cast<lto::optional<lowered_type> const*>(p);                          \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ lifted_type* lift(lowered_type* p)                              \
  {                                                                                          \
    return reinterpret_cast<lifted_type*>(p);                                                \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ lifted_type const* lift(lowered_type const* p)                  \
  {                                                                                          \
    return reinterpret_cast<lifted_type const*>(p);                                          \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ cuda::std::optional<lifted_type>* lift(                         \
    lto::optional<lowered_type>* p)                                                          \
  {                                                                                          \
    return reinterpret_cast<cuda::std::optional<lifted_type>*>(p);                           \
  }                                                                                          \
                                                                                             \
  __device__ __forceinline__ cuda::std::optional<lifted_type> const* lift(                   \
    lto::optional<lowered_type> const* p)                                                    \
  {                                                                                          \
    return reinterpret_cast<cuda::std::optional<lifted_type> const*>(p);                     \
  }

CUDF_LTO_MAP(bool, bool);
CUDF_LTO_MAP(lto::data_type, cudf::data_type);
CUDF_LTO_MAP(lto::int8_t, std::int8_t);
CUDF_LTO_MAP(lto::int16_t, std::int16_t);
CUDF_LTO_MAP(lto::int32_t, std::int32_t);
CUDF_LTO_MAP(lto::int64_t, std::int64_t);
CUDF_LTO_MAP(lto::uint8_t, std::uint8_t);
CUDF_LTO_MAP(lto::uint16_t, std::uint16_t);
CUDF_LTO_MAP(lto::uint32_t, std::uint32_t);
CUDF_LTO_MAP(lto::uint64_t, std::uint64_t);
CUDF_LTO_MAP(lto::float32_t, float);
CUDF_LTO_MAP(lto::float64_t, double);
CUDF_LTO_MAP(lto::decimal32, numeric::decimal32);
CUDF_LTO_MAP(lto::decimal64, numeric::decimal64);
CUDF_LTO_MAP(lto::decimal128, numeric::decimal128);
CUDF_LTO_MAP(lto::string_view, cudf::string_view);
CUDF_LTO_MAP(lto::timestamp_D, cudf::timestamp_D);
CUDF_LTO_MAP(lto::timestamp_h, cudf::timestamp_h);
CUDF_LTO_MAP(lto::timestamp_m, cudf::timestamp_m);
CUDF_LTO_MAP(lto::timestamp_s, cudf::timestamp_s);
CUDF_LTO_MAP(lto::timestamp_ms, cudf::timestamp_ms);
CUDF_LTO_MAP(lto::timestamp_us, cudf::timestamp_us);
CUDF_LTO_MAP(lto::timestamp_ns, cudf::timestamp_ns);
CUDF_LTO_MAP(lto::duration_D, cudf::duration_D);
CUDF_LTO_MAP(lto::duration_h, cudf::duration_h);
CUDF_LTO_MAP(lto::duration_m, cudf::duration_m);
CUDF_LTO_MAP(lto::duration_s, cudf::duration_s);
CUDF_LTO_MAP(lto::duration_ms, cudf::duration_ms);
CUDF_LTO_MAP(lto::duration_us, cudf::duration_us);
CUDF_LTO_MAP(lto::duration_ns, cudf::duration_ns);
CUDF_LTO_MAP(lto::column_device_view_core, cudf::column_device_view_core);
CUDF_LTO_MAP(lto::mutable_column_device_view_core, cudf::mutable_column_device_view_core);

#undef CUDF_LTO_MAP

}  // namespace lto

}  // namespace CUDF_EXPORT cudf
