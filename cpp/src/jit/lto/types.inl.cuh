/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/thunk.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_EXPORT cudf {
namespace lto {

[[nodiscard]] __device__ size_type string_view::size_bytes() const
{
  return lto::lift(this)->size_bytes();
}

[[nodiscard]] __device__ size_type string_view::length() const { return lto::lift(this)->length(); }

[[nodiscard]] __device__ char const* string_view::data() const { return lto::lift(this)->data(); }

[[nodiscard]] __device__ bool string_view::empty() const { return lto::lift(this)->empty(); }

__device__ char_utf8 string_view::operator[](size_type pos) const
{
  return lto::lift(this)->operator[](pos);
}

[[nodiscard]] __device__ size_type string_view::byte_offset(size_type pos) const
{
  return lto::lift(this)->byte_offset(pos);
}

[[nodiscard]] __device__ int string_view::compare(string_view const& str) const
{
  return lto::lift(this)->compare(*lto::lift(&str));
}

__device__ int string_view::compare(char const* str, size_type bytes) const
{
  return lto::lift(this)->compare(str, bytes);
}

__device__ bool string_view::operator==(string_view const& rhs) const
{
  return lto::lift(this)->operator==(*lto::lift(&rhs));
}

__device__ bool string_view::operator!=(string_view const& rhs) const
{
  return lto::lift(this)->operator!=(*lto::lift(&rhs));
}

__device__ bool string_view::operator<(string_view const& rhs) const
{
  return lto::lift(this)->operator<(*lto::lift(&rhs));
}

__device__ bool string_view::operator>(string_view const& rhs) const
{
  return lto::lift(this)->operator>(*lto::lift(&rhs));
}

__device__ bool string_view::operator<=(string_view const& rhs) const
{
  return lto::lift(this)->operator<=(*lto::lift(&rhs));
}

__device__ bool string_view::operator>=(string_view const& rhs) const
{
  return lto::lift(this)->operator>=(*lto::lift(&rhs));
}

[[nodiscard]] __device__ size_type string_view::find(string_view const& str,
                                                     size_type pos,
                                                     size_type count) const
{
  return lto::lift(this)->find(*lto::lift(&str), pos, count);
}

__device__ size_type string_view::find(char const* str,
                                       size_type bytes,
                                       size_type pos,
                                       size_type count) const
{
  return lto::lift(this)->find(str, bytes, pos, count);
}

[[nodiscard]] __device__ size_type string_view::find(char_utf8 character,
                                                     size_type pos,
                                                     size_type count) const
{
  return lto::lift(this)->find(character, pos, count);
}

[[nodiscard]] __device__ size_type string_view::rfind(string_view const& str,
                                                      size_type pos,
                                                      size_type count) const
{
  return lto::lift(this)->rfind(*lto::lift(&str), pos, count);
}

__device__ size_type string_view::rfind(char const* str,
                                        size_type bytes,
                                        size_type pos,
                                        size_type count) const
{
  return lto::lift(this)->rfind(str, bytes, pos, count);
}

[[nodiscard]] __device__ size_type string_view::rfind(char_utf8 character,
                                                      size_type pos,
                                                      size_type count) const
{
  return lto::lift(this)->rfind(character, pos, count);
}

[[nodiscard]] __device__ string_view string_view::substr(size_type start, size_type length) const
{
  auto ret = lto::lift(this)->substr(start, length);
  return *lto::lower(&ret);
}

template <typename T>
__device__ T const* column_device_view_core::head() const
{
  return lto::lower(lto::lift(this)->head<lto::lifted_type_of<T>>());
}

#define CUDF_LTO_INST(Type) \
  template __device__ Type const* column_device_view_core::head<Type>() const

CUDF_LTO_INST(bool);
CUDF_LTO_INST(int8_t);
CUDF_LTO_INST(int16_t);
CUDF_LTO_INST(int32_t);
CUDF_LTO_INST(int64_t);
CUDF_LTO_INST(uint8_t);
CUDF_LTO_INST(uint16_t);
CUDF_LTO_INST(uint32_t);
CUDF_LTO_INST(uint64_t);
CUDF_LTO_INST(float32_t);
CUDF_LTO_INST(float64_t);
CUDF_LTO_INST(timestamp_D);
CUDF_LTO_INST(timestamp_h);
CUDF_LTO_INST(timestamp_m);
CUDF_LTO_INST(timestamp_s);
CUDF_LTO_INST(timestamp_ms);
CUDF_LTO_INST(timestamp_us);
CUDF_LTO_INST(timestamp_ns);
CUDF_LTO_INST(duration_D);
CUDF_LTO_INST(duration_h);
CUDF_LTO_INST(duration_m);
CUDF_LTO_INST(duration_s);
CUDF_LTO_INST(duration_ms);
CUDF_LTO_INST(duration_us);
CUDF_LTO_INST(duration_ns);

#undef CUDF_LTO_INST

__device__ size_type column_device_view_core::size() const { return lto::lift(this)->size(); }

__device__ bool column_device_view_core::nullable() const { return lto::lift(this)->nullable(); }

__device__ bitmask_type const* column_device_view_core::null_mask() const
{
  return lto::lift(this)->null_mask();
}

__device__ size_type column_device_view_core::offset() const { return lto::lift(this)->offset(); }

__device__ bool column_device_view_core::is_valid(size_type index) const
{
  return lto::lift(this)->is_valid(index);
}

__device__ bool column_device_view_core::is_valid_nocheck(size_type index) const
{
  return lto::lift(this)->is_valid_nocheck(index);
}

__device__ bool column_device_view_core::is_null(size_type index) const
{
  return lto::lift(this)->is_null(index);
}

__device__ bool column_device_view_core::is_null_nocheck(size_type index) const
{
  return lto::lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type column_device_view_core::get_mask_word(size_type index) const
{
  return lto::lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T column_device_view_core::element(size_type index) const
{
  auto ret = lto::lift(this)->element<lto::lifted_type_of<T>>(index);
  return *lto::lower(&ret);
}

#define CUDF_LTO_INST(Type) \
  template __device__ Type column_device_view_core::element<Type>(size_type idx) const

CUDF_LTO_INST(bool);
CUDF_LTO_INST(int8_t);
CUDF_LTO_INST(int16_t);
CUDF_LTO_INST(int32_t);
CUDF_LTO_INST(int64_t);
CUDF_LTO_INST(uint8_t);
CUDF_LTO_INST(uint16_t);
CUDF_LTO_INST(uint32_t);
CUDF_LTO_INST(uint64_t);
CUDF_LTO_INST(decimal32);
CUDF_LTO_INST(decimal64);
CUDF_LTO_INST(decimal128);
CUDF_LTO_INST(float32_t);
CUDF_LTO_INST(float64_t);
CUDF_LTO_INST(string_view);
CUDF_LTO_INST(timestamp_D);
CUDF_LTO_INST(timestamp_h);
CUDF_LTO_INST(timestamp_m);
CUDF_LTO_INST(timestamp_s);
CUDF_LTO_INST(timestamp_ms);
CUDF_LTO_INST(timestamp_us);
CUDF_LTO_INST(timestamp_ns);
CUDF_LTO_INST(duration_D);
CUDF_LTO_INST(duration_h);
CUDF_LTO_INST(duration_m);
CUDF_LTO_INST(duration_s);
CUDF_LTO_INST(duration_ms);
CUDF_LTO_INST(duration_us);
CUDF_LTO_INST(duration_ns);

#undef CUDF_LTO_INST

__device__ size_type column_device_view_core::num_child_columns() const
{
  return lto::lift(this)->num_child_columns();
}

template <typename T>
__device__ T* mutable_column_device_view_core::head() const
{
  return lto::lower(lto::lift(this)->head<lto::lifted_type_of<T>>());
}

#define CUDF_LTO_INST(Type) \
  template __device__ Type* mutable_column_device_view_core::head<Type>() const

CUDF_LTO_INST(bool);
CUDF_LTO_INST(int8_t);
CUDF_LTO_INST(int16_t);
CUDF_LTO_INST(int32_t);
CUDF_LTO_INST(int64_t);
CUDF_LTO_INST(uint8_t);
CUDF_LTO_INST(uint16_t);
CUDF_LTO_INST(uint32_t);
CUDF_LTO_INST(uint64_t);
CUDF_LTO_INST(float32_t);
CUDF_LTO_INST(float64_t);
CUDF_LTO_INST(timestamp_D);
CUDF_LTO_INST(timestamp_h);
CUDF_LTO_INST(timestamp_m);
CUDF_LTO_INST(timestamp_s);
CUDF_LTO_INST(timestamp_ms);
CUDF_LTO_INST(timestamp_us);
CUDF_LTO_INST(timestamp_ns);
CUDF_LTO_INST(duration_D);
CUDF_LTO_INST(duration_h);
CUDF_LTO_INST(duration_m);
CUDF_LTO_INST(duration_s);
CUDF_LTO_INST(duration_ms);
CUDF_LTO_INST(duration_us);
CUDF_LTO_INST(duration_ns);

#undef CUDF_LTO_INST

__device__ size_type mutable_column_device_view_core::size() const
{
  return lto::lift(this)->size();
}

__device__ bool mutable_column_device_view_core::nullable() const
{
  return lto::lift(this)->nullable();
}

__device__ bitmask_type* mutable_column_device_view_core::null_mask() const
{
  return lto::lift(this)->null_mask();
}

__device__ size_type mutable_column_device_view_core::offset() const
{
  return lto::lift(this)->offset();
}

__device__ bool mutable_column_device_view_core::is_valid(size_type index) const
{
  return lto::lift(this)->is_valid(index);
}

__device__ bool mutable_column_device_view_core::is_valid_nocheck(size_type index) const
{
  return lto::lift(this)->is_valid_nocheck(index);
}

__device__ bool mutable_column_device_view_core::is_null(size_type index) const
{
  return lto::lift(this)->is_null(index);
}

__device__ bool mutable_column_device_view_core::is_null_nocheck(size_type index) const
{
  return lto::lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type mutable_column_device_view_core::get_mask_word(size_type index) const
{
  return lto::lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T mutable_column_device_view_core::element(size_type index) const
{
  auto ret = lto::lift(this)->element<lto::lifted_type_of<T>>(index);
  return *lto::lower(&ret);
}

#define CUDF_LTO_INST(Type) \
  template __device__ Type mutable_column_device_view_core::element<Type>(size_type idx) const

CUDF_LTO_INST(bool);
CUDF_LTO_INST(int8_t);
CUDF_LTO_INST(int16_t);
CUDF_LTO_INST(int32_t);
CUDF_LTO_INST(int64_t);
CUDF_LTO_INST(uint8_t);
CUDF_LTO_INST(uint16_t);
CUDF_LTO_INST(uint32_t);
CUDF_LTO_INST(uint64_t);
CUDF_LTO_INST(float32_t);
CUDF_LTO_INST(float64_t);
CUDF_LTO_INST(timestamp_D);
CUDF_LTO_INST(timestamp_h);
CUDF_LTO_INST(timestamp_m);
CUDF_LTO_INST(timestamp_s);
CUDF_LTO_INST(timestamp_ms);
CUDF_LTO_INST(timestamp_us);
CUDF_LTO_INST(timestamp_ns);
CUDF_LTO_INST(duration_D);
CUDF_LTO_INST(duration_h);
CUDF_LTO_INST(duration_m);
CUDF_LTO_INST(duration_s);
CUDF_LTO_INST(duration_ms);
CUDF_LTO_INST(duration_us);
CUDF_LTO_INST(duration_ns);

#undef CUDF_LTO_INST

}  // namespace lto
}  // namespace CUDF_EXPORT cudf
