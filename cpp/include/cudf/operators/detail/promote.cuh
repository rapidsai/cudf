/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {
namespace ops::detail {

/**
 * @brief Type promotion map used for overflow-safe integral arithmetic.
 *
 * @tparam T Input integral type.
 */
template <typename T>
struct promoted_t;

template <>
struct promoted_t<int8_t> {
  using type = int16_t;
};

template <>
struct promoted_t<uint8_t> {
  using type = uint16_t;
};

template <>
struct promoted_t<int16_t> {
  using type = int32_t;
};

template <>
struct promoted_t<uint16_t> {
  using type = uint32_t;
};

template <>
struct promoted_t<int32_t> {
  using type = int64_t;
};

template <>
struct promoted_t<uint32_t> {
  using type = uint64_t;
};

template <>
struct promoted_t<int64_t> {
  using type = __int128;
};

template <>
struct promoted_t<uint64_t> {
  using type = unsigned __int128;
};

template <typename T>
using promote = typename promoted_t<T>::type;

}  // namespace ops::detail
}  // namespace CUDF_EXPORT cudf
