/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {

struct nullopt_t {};

inline constexpr nullopt_t nullopt;

template <typename T>
struct optional {
  T _value = {};

  bool _is_valid = false;

  constexpr optional() = default;

  __device__ constexpr optional(nullopt_t) {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : _value{static_cast<Args&&>(args)...}, _is_valid{true}
  {
  }

  __device__ constexpr optional(T value) : _value{value}, _is_valid{true} {}

  __device__ constexpr bool is_valid() const { return _is_valid; }

  __device__ constexpr bool is_null() const { return !_is_valid; }

  __device__ constexpr void reset() { _is_valid = false; }

  __device__ constexpr T const& get() const { return _value; }

  __device__ constexpr T& get() { return _value; }

  __device__ constexpr T const* operator->() const { return &_value; }

  __device__ constexpr T* operator->() { return &_value; }

  __device__ constexpr T const& operator*() const { return _value; }

  __device__ constexpr T& operator*() { return _value; }

  __device__ constexpr T const& value() const { return _value; }

  __device__ constexpr T& value() { return _value; }

  __device__ constexpr explicit operator bool() const { return _is_valid; }

  __device__ constexpr T value_or(T v) const { return _is_valid ? _value : v; }
};

template <typename T>
optional(T) -> optional<T>;

}  // namespace CUDF_EXPORT cudf
