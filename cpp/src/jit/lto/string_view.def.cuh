/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/string_view.cuh>
#include <cudf/jit/lto/thunk.cuh>

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

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
