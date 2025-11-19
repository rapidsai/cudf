/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Formats a column view as a string
 *
 * @param col The input column view
 * @param delimiter The delimiter to put between strings
 */
std::string to_string(cudf::column_view const& col, std::string const& delimiter);

/**
 * @brief Convert column values to a host vector of strings
 *
 * @param col The input column view
 */
std::vector<std::string> to_strings(cudf::column_view const& col);

/**
 * @brief Print a column view to an ostream
 *
 * @param col The input column view
 * @param os The output stream
 */
void print(cudf::column_view const& col, std::ostream& os = std::cout);

}  // namespace test
}  // namespace CUDF_EXPORT cudf
