/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <string>

namespace cudf {
namespace jit {
/**
 * @brief Get the raw pointer to data in a (mutable_)column_view
 */
void const* get_data_ptr(column_view const& view);

/**
 * @brief Get the raw pointer to data in a scalar
 */
void const* get_data_ptr(scalar const& s);

}  // namespace jit
}  // namespace cudf
