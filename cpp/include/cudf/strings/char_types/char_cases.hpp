/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
