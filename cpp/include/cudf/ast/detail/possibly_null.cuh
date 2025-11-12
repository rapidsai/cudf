/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {

namespace ast::detail {

// Type trait for wrapping nullable types in a cuda::std::optional. Non-nullable
// types are returned as is.
template <typename T, bool has_nulls>
struct possibly_null_value;

template <typename T>
struct possibly_null_value<T, true> {
  using type = cuda::std::optional<T>;
};

template <typename T>
struct possibly_null_value<T, false> {
  using type = T;
};

template <typename T, bool has_nulls>
using possibly_null_value_t = typename possibly_null_value<T, has_nulls>::type;

}  // namespace ast::detail
}  // namespace CUDF_EXPORT cudf
