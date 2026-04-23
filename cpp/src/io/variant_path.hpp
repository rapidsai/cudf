/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief One step in a parsed VARIANT path: either a dictionary field name or an array index.
 *
 * String alternative (`std::string`): descend into an object via the key with that name.
 * Integer alternative (`cudf::size_type`): descend into an array via that non-negative index.
 */
using variant_path_step = std::variant<std::string, cudf::size_type>;

/**
 * @brief Parse a JSONPath-like VARIANT path string into an ordered sequence of steps.
 *
 * Grammar (subset of JSONPath; no filters/expressions):
 *   path   := "$"? first_step step*
 *   step   := "." name | "[" index "]" | "[" quoted "]"
 *   name   := [A-Za-z_][A-Za-z0-9_]*   (first step may also be a bare name)
 *   quoted := "'...'" | "\"...\""   (no escape handling; may not contain the wrapping quote char)
 *   index  := non-negative base-10 integer
 *
 * @throws std::invalid_argument on empty path, `[*]` wildcard, negative index, or malformed syntax
 */
std::vector<variant_path_step> parse_variant_path(std::string_view path);

}  // namespace cudf::io::parquet::detail
