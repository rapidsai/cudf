/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Converts UTF-8 encoded strings to CP932 (Shift-JIS) encoding.
 *
 * This function converts UTF-8 strings to CP932 encoding, also known as
 * Windows code page 932 or Windows-31J. CP932 is commonly used for
 * Japanese text on Windows systems.
 *
 * Encoding rules:
 * - ASCII characters (U+0000-U+007F): 1 byte, same as UTF-8
 * - Half-width katakana (U+FF61-U+FF9F): 1 byte (0xA1-0xDF)
 * - JIS X 0208 characters (kanji, hiragana, full-width symbols): 2 bytes
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if any character cannot be represented in CP932 encoding
 *
 * @param input Strings column containing UTF-8 encoded text
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with CP932 encoded strings
 */
std::unique_ptr<column> utf8_to_cp932(
  strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
