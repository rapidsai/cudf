/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/concepts>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

template <typename T>
concept unsigned_integer =
  cudf::concepts::integral_not_bool<T> && cuda::std::is_unsigned_v<cuda::std::remove_cv_t<T>>;

template <typename T>
constexpr bool is_nullable = false;

template <typename T>
constexpr bool is_nullable<cuda::std::optional<T>> = true;

template <typename T>
concept nullable = is_nullable<T>;

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
