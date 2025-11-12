/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
namespace detail {
/**
 * @brief Regenerates the special case mapping tables used to handle non-trivial unicode
 * character case conversions.
 *
 * 'special' cased characters are those defined as not having trivial single->single character
 * mappings when having upper(), lower() or titlecase() operations applied.  Typically this is
 * for cases where a single character maps to multiple, but there are also cases of
 * non-reversible mappings, where:  codepoint != lower(upper(code_point)).
 */
void generate_special_mapping_hash_table();

}  // namespace detail
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
