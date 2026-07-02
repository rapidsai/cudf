/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_slice
 * @{
 * @file
 */

/**
 * @brief Indicates whether the start position is interpreted as a zero-based or one-based index.
 */
enum class start_indexing_policy {
  ZERO_BASED,  ///< The first character position is 0
  ONE_BASED    ///< The first character position is 1; start value 0 maps to the first character
};

/**
 * @brief Indicates how a negative start position is interpreted.
 */
enum class negative_start_policy {
  CLAMP_TO_ZERO,   ///< Negative start positions are clamped to the first character
  RELATIVE_TO_END  ///< Negative start positions are relative to the end of the string
};

/**
 * @brief Indicates how the second slicing parameter is interpreted.
 */
enum class slice_bounds_policy {
  START_AND_STOP,   ///< The second parameter is the exclusive stop position
  START_AND_LENGTH  ///< The second parameter is the number of characters to include
};

/**
 * @brief Indicates how negative length values are interpreted.
 */
enum class negative_length_policy {
  PRESERVE,      ///< Negative lengths are preserved when computing the stop position
  CLAMP_TO_ZERO  ///< Negative lengths are clamped to zero
};

/**
 * @brief Indicates how a missing length is interpreted when using START_AND_LENGTH bounds.
 */
enum class missing_length_policy {
  TO_END,   ///< A missing length slices through the end of each string
  MAX_SIZE  ///< A missing length is treated as the maximum supported size_type value
};

/**
 * @brief Options controlling how slice bounds are interpreted.
 */
struct slice_strings_options {
  start_indexing_policy start_indexing{start_indexing_policy::ZERO_BASED};
  negative_start_policy negative_start{negative_start_policy::CLAMP_TO_ZERO};
  slice_bounds_policy bounds{slice_bounds_policy::START_AND_STOP};
  negative_length_policy negative_length{negative_length_policy::PRESERVE};
  missing_length_policy missing_length{missing_length_policy::TO_END};
};

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions to retrieve in each string are `[start,stop)`.
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r = slice_strings(s,2,6)
 * r is now ["llo","odby"]
 * r2 = slice_strings(s,2,5,2)
 * r2 is now ["lo","ob"]
 * @endcode
 *
 * @param input Strings column for this operation
 * @param start First character position to begin the substring
 * @param stop Last character position (exclusive) to end the substring
 * @param step Distance between input characters retrieved
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sorted elements of this instance
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions are interpreted according to the provided options.
 * The default options preserve the zero-based `[start, stop)` interpretation.
 *
 * @param input Strings column for this operation
 * @param start First character position to begin the substring
 * @param stop Last character position (exclusive) or length, depending on `options`
 * @param step Distance between input characters retrieved. Must be 1 unless `options` is the
 * default zero-based `[start, stop)` behavior.
 * @param options Options controlling how slice bounds are interpreted
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sliced strings
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  numeric_scalar<size_type> const& start,
  numeric_scalar<size_type> const& stop,
  numeric_scalar<size_type> const& step,
  slice_strings_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique ranges for each string.
 *
 * The character positions to retrieve in each string are specified in
 * the `starts` and `stops` integer columns.
 * If a start position is outside a string's length, an empty
 * string is returned for that entry. If a stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string. Any stop position value set to -1 will
 * indicate to use the end of the string as the stop position for that
 * string.
 *
 * Null string entries will return null output string entries.
 *
 * The starts and stops column must both be the same integer type and
 * must be the same size as the strings column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = slice_strings(s,starts,stops)
 * r is now ["ello","od"]
 * @endcode
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param input Strings column for this operation
 * @param starts First character positions to begin the substring
 * @param stops Last character (exclusive) positions to end the substring
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sorted elements of this instance
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  column_view const& starts,
  column_view const& stops,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique bounds for each string.
 *
 * The character positions are interpreted according to the provided options.
 * The default options preserve the zero-based `[start, stop)` interpretation.
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param input Strings column for this operation
 * @param starts First character positions to begin the substring
 * @param stops Last character position (exclusive) or length, depending on `options`
 * @param options Options controlling how slice bounds are interpreted
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with sliced strings
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& input,
  column_view const& starts,
  column_view const& stops,
  slice_strings_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
