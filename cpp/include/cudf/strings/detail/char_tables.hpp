/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace CUDF_EXPORT cudf {
namespace strings::detail {
// Type for the character flags table.
using character_flags_table_type = std::uint8_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * characters flags table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to check the type of character like
 * alphanumeric, decimal, etc.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Device memory pointer to character flags table.
 */
character_flags_table_type const* get_character_flags_table(
  rmm::cuda_stream_view stream = cudf::get_default_stream());

// utilities to dissect a character-table flag
constexpr uint8_t IS_DECIMAL(uint8_t x) { return ((x) & (1 << 0)); }
constexpr uint8_t IS_NUMERIC(uint8_t x) { return ((x) & (1 << 1)); }
constexpr uint8_t IS_DIGIT(uint8_t x) { return ((x) & (1 << 2)); }
constexpr uint8_t IS_ALPHA(uint8_t x) { return ((x) & (1 << 3)); }
constexpr uint8_t IS_SPACE(uint8_t x) { return ((x) & (1 << 4)); }
constexpr uint8_t IS_UPPER(uint8_t x) { return ((x) & (1 << 5)); }
constexpr uint8_t IS_LOWER(uint8_t x) { return ((x) & (1 << 6)); }
constexpr uint8_t IS_SPECIAL(uint8_t x) { return ((x) & (1 << 7)); }
constexpr uint8_t IS_ALPHANUM(uint8_t x) { return ((x) & (0x0F)); }
constexpr uint8_t IS_UPPER_OR_LOWER(uint8_t x) { return ((x) & ((1 << 5) | (1 << 6))); }
constexpr uint8_t ALL_FLAGS = 0xFF;

// Type for the character cases table.
using character_cases_table_type = uint16_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * characters case table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to map upper and lower case characters with
 * their counterpart.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Device memory pointer to character cases table.
 */
character_cases_table_type const* get_character_cases_table(
  rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Case mapping structure for special characters.
 *
 * This is used for special mapping of a small set of characters that do not
 * fit in the character-cases-table.
 *
 * @see cpp/src/strings/char_types/char_cases.h
 */
struct special_case_mapping {
  uint16_t num_upper_chars;
  uint16_t upper[3];  // NOLINT
  uint16_t num_lower_chars;
  uint16_t lower[3];  // NOLINT
};

/**
 * @brief Returns pointer to device memory that contains the special
 * case mapping table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used to handle special case character mappings that
 * don't trivially work with the normal character cases table.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Device memory pointer to the special case mapping table
 */
special_case_mapping const* get_special_case_mapping_table(
  rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Get the special mapping table index for a given code-point.
 *
 * @see cpp/src/strings/char_types/char_cases.h
 */
constexpr uint16_t get_special_case_hash_index(uint32_t code_point)
{
  constexpr uint16_t special_case_prime = 499;  // computed from generate_special_mapping_hash_table
  return static_cast<uint16_t>(code_point % special_case_prime);
}

}  // namespace strings::detail
}  // namespace CUDF_EXPORT cudf
