/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/thunk.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

[[nodiscard]] __device__ size_type string_view::size_bytes() const
{
  return lift(this)->size_bytes();
}

[[nodiscard]] __device__ size_type string_view::length() const { return lift(this)->length(); }

[[nodiscard]] __device__ char const* string_view::data() const { return lift(this)->data(); }

[[nodiscard]] __device__ bool string_view::empty() const { return lift(this)->empty(); }

__device__ char_utf8 string_view::operator[](size_type pos) const
{
  return lift(this)->operator[](pos);
}

[[nodiscard]] __device__ size_type string_view::byte_offset(size_type pos) const
{
  return lift(this)->byte_offset(pos);
}

[[nodiscard]] __device__ int string_view::compare(string_view const& str) const
{
  return lift(this)->compare(*lift(&str));
}

__device__ int string_view::compare(char const* str, size_type bytes) const
{
  return lift(this)->compare(str, bytes);
}

__device__ bool string_view::operator==(string_view const& rhs) const
{
  return lift(this)->operator==(*lift(&rhs));
}

__device__ bool string_view::operator!=(string_view const& rhs) const
{
  return lift(this)->operator!=(*lift(&rhs));
}

__device__ bool string_view::operator<(string_view const& rhs) const
{
  return lift(this)->operator<(*lift(&rhs));
}

__device__ bool string_view::operator>(string_view const& rhs) const
{
  return lift(this)->operator>(*lift(&rhs));
}

__device__ bool string_view::operator<=(string_view const& rhs) const
{
  return lift(this)->operator<=(*lift(&rhs));
}

__device__ bool string_view::operator>=(string_view const& rhs) const
{
  return lift(this)->operator>=(*lift(&rhs));
}

[[nodiscard]] __device__ size_type string_view::find(string_view const& str,
                                                     size_type pos,
                                                     size_type count) const
{
  return lift(this)->find(*lift(&str), pos, count);
}

__device__ size_type string_view::find(char const* str,
                                       size_type bytes,
                                       size_type pos,
                                       size_type count) const
{
  return lift(this)->find(str, bytes, pos, count);
}

[[nodiscard]] __device__ size_type string_view::find(char_utf8 character,
                                                     size_type pos,
                                                     size_type count) const
{
  return lift(this)->find(character, pos, count);
}

[[nodiscard]] __device__ size_type string_view::rfind(string_view const& str,
                                                      size_type pos,
                                                      size_type count) const
{
  return lift(this)->rfind(*lift(&str), pos, count);
}

__device__ size_type string_view::rfind(char const* str,
                                        size_type bytes,
                                        size_type pos,
                                        size_type count) const
{
  return lift(this)->rfind(str, bytes, pos, count);
}

[[nodiscard]] __device__ size_type string_view::rfind(char_utf8 character,
                                                      size_type pos,
                                                      size_type count) const
{
  return lift(this)->rfind(character, pos, count);
}

[[nodiscard]] __device__ string_view string_view::substr(size_type start, size_type length) const
{
  auto ret = lift(this)->substr(start, length);
  return *lower(&ret);
}

template <typename T>
__device__ T const* column_device_view_core::head() const
{
  return lower(lift(this)->head<lifted_type_of<T>>());
}

__device__ size_type column_device_view_core::size() const { return lift(this)->size(); }

__device__ bool column_device_view_core::nullable() const { return lift(this)->nullable(); }

__device__ bitmask_type const* column_device_view_core::null_mask() const
{
  return lift(this)->null_mask();
}

__device__ size_type column_device_view_core::offset() const { return lift(this)->offset(); }

__device__ bool column_device_view_core::is_valid(size_type index) const
{
  return lift(this)->is_valid(index);
}

__device__ bool column_device_view_core::is_valid_nocheck(size_type index) const
{
  return lift(this)->is_valid_nocheck(index);
}

__device__ bool column_device_view_core::is_null(size_type index) const
{
  return lift(this)->is_null(index);
}

__device__ bool column_device_view_core::is_null_nocheck(size_type index) const
{
  return lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type column_device_view_core::get_mask_word(size_type index) const
{
  return lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T column_device_view_core::element(size_type index) const
{
  auto ret = lift(this)->element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ optional<T> column_device_view_core::nullable_element(size_type index) const
{
  auto ret = lift(this)->nullable_element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

__device__ size_type column_device_view_core::num_child_columns() const
{
  return lift(this)->num_child_columns();
}

template <typename T>
__device__ T* mutable_column_device_view_core::head() const
{
  return lower(lift(this)->head<lifted_type_of<T>>());
}

__device__ size_type mutable_column_device_view_core::size() const { return lift(this)->size(); }

__device__ bool mutable_column_device_view_core::nullable() const { return lift(this)->nullable(); }

__device__ bitmask_type* mutable_column_device_view_core::null_mask() const
{
  return lift(this)->null_mask();
}

__device__ size_type mutable_column_device_view_core::offset() const
{
  return lift(this)->offset();
}

__device__ bool mutable_column_device_view_core::is_valid(size_type index) const
{
  return lift(this)->is_valid(index);
}

__device__ bool mutable_column_device_view_core::is_valid_nocheck(size_type index) const
{
  return lift(this)->is_valid_nocheck(index);
}

__device__ bool mutable_column_device_view_core::is_null(size_type index) const
{
  return lift(this)->is_null(index);
}

__device__ bool mutable_column_device_view_core::is_null_nocheck(size_type index) const
{
  return lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type mutable_column_device_view_core::get_mask_word(size_type index) const
{
  return lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T mutable_column_device_view_core::element(size_type index) const
{
  auto ret = lift(this)->element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ optional<T> mutable_column_device_view_core::nullable_element(size_type index) const
{
  auto ret = lift(this)->nullable_element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ void mutable_column_device_view_core::assign(size_type index, T value) const
{
  lift(this)->assign<lifted_type_of<T>>(index, *lift(&value));
}

/// Explicit template instantiations

#define DO_IT(T) template __device__ T const* column_device_view_core::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) template __device__ T column_device_view_core::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) \
  template __device__ optional<T> column_device_view_core::nullable_element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) template __device__ T* mutable_column_device_view_core::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) \
  template __device__ T mutable_column_device_view_core::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                        \
  template __device__ optional<T> mutable_column_device_view_core::nullable_element<T>( \
    size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) \
  template __device__ void mutable_column_device_view_core::assign<T>(size_type idx, T value) const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
