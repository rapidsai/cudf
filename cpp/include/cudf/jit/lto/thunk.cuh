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

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

template <typename T>
struct lowered_type_of_t;

template <typename T>
using lowered_type_of = typename lowered_type_of_t<T>::type;

template <typename T>
struct lifted_type_of_t;

template <typename T>
using lifted_type_of = typename lifted_type_of_t<T>::type;

#define CUDF_LTO_MAP(lowered_type, lifted_type)                                                 \
  static_assert(sizeof(lowered_type) == sizeof(lifted_type),                                    \
                "(1: size). Lowered and Lifted types must be bitwise-equivalent");              \
  static_assert(alignof(lowered_type) == alignof(lifted_type),                                  \
                "(2: alignment). Lowered and Lifted types must be bitwise-equivalent");         \
  static_assert(                                                                                \
    sizeof(optional<lowered_type>) == sizeof(cuda::std::optional<lifted_type>),                 \
    "(1: size). Lowered and Lifted types must have bitwise-equivalent optional types");         \
  static_assert(                                                                                \
    alignof(optional<lowered_type>) == alignof(cuda::std::optional<lifted_type>),               \
    "(2: alignment). Lowered and Lifted types must have bitwise-equivalent optional types");    \
                                                                                                \
  template <>                                                                                   \
  struct lifted_type_of_t<lowered_type> {                                                       \
    using type = lifted_type;                                                                   \
  };                                                                                            \
                                                                                                \
  template <>                                                                                   \
  struct lowered_type_of_t<lifted_type> {                                                       \
    using type = lowered_type;                                                                  \
  };                                                                                            \
                                                                                                \
  __device__ __forceinline__ lowered_type* lower(lifted_type* p)                                \
  {                                                                                             \
    return reinterpret_cast<lowered_type*>(p);                                                  \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ lowered_type const* lower(lifted_type const* p)                    \
  {                                                                                             \
    return reinterpret_cast<lowered_type const*>(p);                                            \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ optional<lowered_type>* lower(cuda::std::optional<lifted_type>* p) \
  {                                                                                             \
    return reinterpret_cast<optional<lowered_type>*>(p);                                        \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ optional<lowered_type> const* lower(                               \
    cuda::std::optional<lifted_type> const* p)                                                  \
  {                                                                                             \
    return reinterpret_cast<optional<lowered_type> const*>(p);                                  \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ lifted_type* lift(lowered_type* p)                                 \
  {                                                                                             \
    return reinterpret_cast<lifted_type*>(p);                                                   \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ lifted_type const* lift(lowered_type const* p)                     \
  {                                                                                             \
    return reinterpret_cast<lifted_type const*>(p);                                             \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ cuda::std::optional<lifted_type>* lift(optional<lowered_type>* p)  \
  {                                                                                             \
    return reinterpret_cast<cuda::std::optional<lifted_type>*>(p);                              \
  }                                                                                             \
                                                                                                \
  __device__ __forceinline__ cuda::std::optional<lifted_type> const* lift(                      \
    optional<lowered_type> const* p)                                                            \
  {                                                                                             \
    return reinterpret_cast<cuda::std::optional<lifted_type> const*>(p);                        \
  }

CUDF_LTO_MAP(bool, bool);
CUDF_LTO_MAP(data_type, cudf::data_type);
CUDF_LTO_MAP(int8_t, std::int8_t);
CUDF_LTO_MAP(int16_t, std::int16_t);
CUDF_LTO_MAP(int32_t, std::int32_t);
CUDF_LTO_MAP(int64_t, std::int64_t);
CUDF_LTO_MAP(uint8_t, std::uint8_t);
CUDF_LTO_MAP(uint16_t, std::uint16_t);
CUDF_LTO_MAP(uint32_t, std::uint32_t);
CUDF_LTO_MAP(uint64_t, std::uint64_t);
CUDF_LTO_MAP(float32_t, float);
CUDF_LTO_MAP(float64_t, double);
CUDF_LTO_MAP(decimal32, numeric::decimal32);
CUDF_LTO_MAP(decimal64, numeric::decimal64);
CUDF_LTO_MAP(decimal128, numeric::decimal128);
CUDF_LTO_MAP(string_view, cudf::string_view);
CUDF_LTO_MAP(timestamp_D, cudf::timestamp_D);
CUDF_LTO_MAP(timestamp_h, cudf::timestamp_h);
CUDF_LTO_MAP(timestamp_m, cudf::timestamp_m);
CUDF_LTO_MAP(timestamp_s, cudf::timestamp_s);
CUDF_LTO_MAP(timestamp_ms, cudf::timestamp_ms);
CUDF_LTO_MAP(timestamp_us, cudf::timestamp_us);
CUDF_LTO_MAP(timestamp_ns, cudf::timestamp_ns);
CUDF_LTO_MAP(duration_D, cudf::duration_D);
CUDF_LTO_MAP(duration_h, cudf::duration_h);
CUDF_LTO_MAP(duration_m, cudf::duration_m);
CUDF_LTO_MAP(duration_s, cudf::duration_s);
CUDF_LTO_MAP(duration_ms, cudf::duration_ms);
CUDF_LTO_MAP(duration_us, cudf::duration_us);
CUDF_LTO_MAP(duration_ns, cudf::duration_ns);
CUDF_LTO_MAP(column_view, cudf::column_device_view_core);
CUDF_LTO_MAP(mutable_column_view, cudf::mutable_column_device_view_core);

#undef CUDF_LTO_MAP

}  // namespace lto

}  // namespace CUDF_LTO_EXPORT cudf
