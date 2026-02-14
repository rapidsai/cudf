/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/export.cuh>
#include <cudf/jit/lto/optional.cuh>
#include <cudf/jit/lto/string_view.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

struct alignas(16) CUDF_LTO_ALIAS column_view {
 private:
  data_type _type                      = {};
  size_type _size                      = 0;
  void const* _data                    = nullptr;
  bitmask_type const* _null_mask       = nullptr;
  size_type _offset                    = 0;
  column_view* _d_children = nullptr;
  size_type _num_children              = 0;

 public:
  template <typename T>
  __device__ T const* head() const;

  __device__ size_type size() const;

  __device__ bool nullable() const;

  __device__ bitmask_type const* null_mask() const;

  __device__ size_type offset() const;

  __device__ bool is_valid(size_type idx) const;

  __device__ bool is_valid_nocheck(size_type idx) const;

  __device__ bool is_null(size_type idx) const;

  __device__ bool is_null_nocheck(size_type idx) const;

  __device__ bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ T element(size_type idx) const;

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const;

  __device__ size_type num_child_columns() const;
};

struct alignas(16) CUDF_LTO_ALIAS mutable_column_view {
 private:
  data_type _type                              = {};
  size_type _size                              = 0;
  void const* _data                            = nullptr;
  bitmask_type const* _null_mask               = nullptr;
  size_type _offset                            = 0;
  mutable_column_view* _d_children = nullptr;
  size_type _num_children                      = 0;

 public:
  template <typename T>
  __device__ T* head() const;

  __device__ size_type size() const;

  __device__ bool nullable() const;

  __device__ bitmask_type* null_mask() const;

  __device__ size_type offset() const;

  __device__ bool is_valid(size_type idx) const;

  __device__ bool is_valid_nocheck(size_type idx) const;

  __device__ bool is_null(size_type idx) const;

  __device__ bool is_null_nocheck(size_type idx) const;

  __device__ bitmask_type get_mask_word(size_type word_index) const;

  template <typename T>
  __device__ T element(size_type idx) const;

  template <typename T>
  __device__ optional<T> nullable_element(size_type idx) const;

  template <typename T>
  __device__ void assign(size_type idx, T value) const;
};

#define FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE \
  DO_IT(bool)                             \
  DO_IT(int8_t)                           \
  DO_IT(int16_t)                          \
  DO_IT(int32_t)                          \
  DO_IT(int64_t)                          \
  DO_IT(uint8_t)                          \
  DO_IT(uint16_t)                         \
  DO_IT(uint32_t)                         \
  DO_IT(uint64_t)                         \
  DO_IT(float32_t)                        \
  DO_IT(float64_t)                        \
  DO_IT(timestamp_D)                      \
  DO_IT(timestamp_h)                      \
  DO_IT(timestamp_m)                      \
  DO_IT(timestamp_s)                      \
  DO_IT(timestamp_ms)                     \
  DO_IT(timestamp_us)                     \
  DO_IT(timestamp_ns)                     \
  DO_IT(duration_D)                       \
  DO_IT(duration_h)                       \
  DO_IT(duration_m)                       \
  DO_IT(duration_s)                       \
  DO_IT(duration_ms)                      \
  DO_IT(duration_us)                      \
  DO_IT(duration_ns)

#define FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE \
  DO_IT(bool)                                \
  DO_IT(int8_t)                              \
  DO_IT(int16_t)                             \
  DO_IT(int32_t)                             \
  DO_IT(int64_t)                             \
  DO_IT(uint8_t)                             \
  DO_IT(uint16_t)                            \
  DO_IT(uint32_t)                            \
  DO_IT(uint64_t)                            \
  DO_IT(decimal32)                           \
  DO_IT(decimal64)                           \
  DO_IT(decimal128)                          \
  DO_IT(float32_t)                           \
  DO_IT(float64_t)                           \
  DO_IT(string_view)                         \
  DO_IT(timestamp_D)                         \
  DO_IT(timestamp_h)                         \
  DO_IT(timestamp_m)                         \
  DO_IT(timestamp_s)                         \
  DO_IT(timestamp_ms)                        \
  DO_IT(timestamp_us)                        \
  DO_IT(timestamp_ns)                        \
  DO_IT(duration_D)                          \
  DO_IT(duration_h)                          \
  DO_IT(duration_m)                          \
  DO_IT(duration_s)                          \
  DO_IT(duration_ms)                         \
  DO_IT(duration_us)                         \
  DO_IT(duration_ns)

#define FOREACH_CUDF_LTO_COLUMN_ASSIGN_TYPE \
  DO_IT(bool)                               \
  DO_IT(int8_t)                             \
  DO_IT(int16_t)                            \
  DO_IT(int32_t)                            \
  DO_IT(int64_t)                            \
  DO_IT(uint8_t)                            \
  DO_IT(uint16_t)                           \
  DO_IT(uint32_t)                           \
  DO_IT(uint64_t)                           \
  DO_IT(decimal32)                          \
  DO_IT(decimal64)                          \
  DO_IT(decimal128)                         \
  DO_IT(float32_t)                          \
  DO_IT(float64_t)                          \
  DO_IT(timestamp_D)                        \
  DO_IT(timestamp_h)                        \
  DO_IT(timestamp_m)                        \
  DO_IT(timestamp_s)                        \
  DO_IT(timestamp_ms)                       \
  DO_IT(timestamp_us)                       \
  DO_IT(timestamp_ns)                       \
  DO_IT(duration_D)                         \
  DO_IT(duration_h)                         \
  DO_IT(duration_m)                         \
  DO_IT(duration_s)                         \
  DO_IT(duration_ms)                        \
  DO_IT(duration_us)                        \
  DO_IT(duration_ns)

#define DO_IT(T) extern template __device__ T const* column_view::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) \
  extern template __device__ T column_view::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                       \
  extern template __device__ optional<T> column_view::nullable_element<T>( \
    size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) extern template __device__ T* mutable_column_view::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) \
  extern template __device__ T mutable_column_view::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                               \
  extern template __device__ optional<T> mutable_column_view::nullable_element<T>( \
    size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                            \
  extern template __device__ void mutable_column_view::assign<T>(size_type idx, \
                                                                             T value) const;

FOREACH_CUDF_LTO_COLUMN_ASSIGN_TYPE

#undef DO_IT

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
