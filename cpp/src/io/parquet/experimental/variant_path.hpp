/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cudf::io::parquet::experimental::detail {

/**
 * @brief One step in a parsed VARIANT path: an object field name to descend into.
 */
using variant_path_step = std::string;

/**
 * @brief Parse a JSONPath-like VARIANT path string into an ordered sequence of object-key steps.
 *
 * Phase-A grammar — object descent only, no array indexing:
 *   path  := "$"? first_step ("." name)*
 *   first := name | "." name
 *   name  := [A-Za-z_][A-Za-z0-9_]*
 *
 * @throws std::invalid_argument on empty path or malformed syntax (including bracket steps,
 *         which are reserved for a later phase that adds array support)
 */
std::vector<variant_path_step> parse_variant_path(std::string_view path);

}  // namespace cudf::io::parquet::experimental::detail
