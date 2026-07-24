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
 * @brief Parse a JSONPath-like VARIANT path string into an ordered sequence of object-key steps.
 *
 * Grammar — object descent only:
 *   path  := "$"? first_step ("." name)*
 *   first := name | "." name
 *   name  := [^.\[]+
 *
 * Names accept any byte except '.' (step separator) and '[' (start of a bracket step,
 * reserved for future array indexing and quoted-name syntax).
 *
 * @throws std::invalid_argument on empty path or malformed syntax (including bracket steps,
 *         which require array-indexing support that is not yet implemented)
 */
[[nodiscard]] std::vector<std::string> parse_variant_path(std::string_view path);

}  // namespace cudf::io::parquet::experimental::detail
