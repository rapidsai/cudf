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

template <typename T>
struct defer {
 private:
  T func_;

 public:
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
defer(T) -> defer<T>;

}  // namespace CUDF_EXPORT cudf