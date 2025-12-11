/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#define CUDF_DEFER__CONCATENATE_DETAIL(x, y) x##y
#define CUDF_DEFER__CONCATENATE(x, y)        CUDF_DEFER__CONCATENATE_DETAIL(x, y)
#define CUDF_DEFER(...)                      ::cudf::defer CUDF_DEFER__CONCATENATE(defer_, __COUNTER__)(__VA_ARGS__)

namespace CUDF_EXPORT cudf {

/// @brief RAII utility to execute a callable at the end of a scope.
/// This is useful for ensuring cleanup code is executed, even in the presence of exceptions.
/// And is intended for wrapping C APIs that require explicit resource management without having to
/// write custom wrapper types.
template <typename T>
struct defer {
 private:
  T func_;

 public:
  /// @brief Construct a `defer` object that will invoke the provided callable upon destruction.
  /// @param args Arguments to forward to the callable's constructor.
  template <typename... Args>
  defer(Args&&... args) : func_{static_cast<Args&&>(args)...}
  {
  }
  defer(defer const&)            = delete;
  defer& operator=(defer const&) = delete;
  defer(defer&&)                 = delete;
  defer& operator=(defer&&)      = delete;
  ~defer() { func_(); }
};

template <typename T>
defer(T) -> defer<T>;  ///< Class template argument deduction guide

}  // namespace CUDF_EXPORT cudf
