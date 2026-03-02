/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/jit/lto/export.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

struct inplace_t {};

inline constexpr inplace_t inplace{};

// TODO: assumes T is trivially copyable
template <typename T>
struct CUDF_LTO_ALIAS optional {
 private:
  T __val;
  bool __engaged;

 public:
  __device__ constexpr optional() : __val{}, __engaged{false} {}

  template <typename... Args>
  __device__ constexpr optional(inplace_t, Args&&... args)
    : __val{static_cast<Args&&>(args)...}, __engaged{true}
  {
  }

  __device__ constexpr optional(T val) : __val{val}, __engaged{true} {}

  constexpr optional(optional const&) = default;

  constexpr optional(optional&&) = default;

  constexpr optional& operator=(optional const&) = default;

  constexpr optional& operator=(optional&&) = default;

  constexpr ~optional() = default;

  __device__ constexpr bool has_value() const { return __engaged; }

  __device__ constexpr void reset() { __engaged = false; }

  __device__ constexpr T const& get() const { return __val; }

  __device__ constexpr T& get() { return __val; }

  __device__ constexpr T const* operator->() const { return &__val; }

  __device__ constexpr T* operator->() { return &__val; }

  __device__ constexpr T const& operator*() const { return __val; }

  __device__ constexpr T& operator*() { return __val; }

  __device__ constexpr T const& value() const { return __val; }

  __device__ constexpr T& value() { return __val; }

  __device__ constexpr explicit operator bool() const { return __engaged; }

  __device__ constexpr T value_or(T __v) const { return __engaged ? __val : __v; }
};

template <typename T>
optional(T) -> optional<T>;

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
