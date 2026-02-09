/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf::detail {

/**
 * @brief Helper class to support inline-overloading for all of a variant's alternative types
 */
template <class... Ts>
struct visitor_overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
visitor_overload(Ts...) -> visitor_overload<Ts...>;

}  // namespace cudf::detail
