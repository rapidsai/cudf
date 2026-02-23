/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cp932_table.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/convert/convert_cp932.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/logical.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Functor for converting UTF-8 strings to CP932 encoding.
 *
 * This follows the two-pass pattern:
 * - Pass 1 (d_chars == nullptr): Calculate output sizes
 * - Pass 2 (d_chars != nullptr): Write the converted characters
 */
struct utf8_to_cp932_fn {
  column_device_view const d_strings;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;
  bool* d_has_errors;  // Flag to track conversion errors

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    string_view const d_str = d_strings.element<string_view>(idx);
    char* out_ptr           = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type nbytes        = 0;

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      char_utf8 const utf8_char = *itr;
      uint32_t const codepoint  = utf8_to_codepoint(utf8_char);

      // Look up CP932 encoding
      int32_t const cp932 = unicode_to_cp932(codepoint);

      if (cp932 < 0) {
        // Character cannot be encoded in CP932
        if (d_has_errors) { *d_has_errors = true; }
        // Skip this character (don't add to output)
        continue;
      }

      if (cp932 < 0x100) {
        // Single-byte character
        nbytes += 1;
        if (out_ptr) { *out_ptr++ = static_cast<char>(cp932); }
      } else {
        // Double-byte character (high byte first)
        nbytes += 2;
        if (out_ptr) {
          *out_ptr++ = static_cast<char>((cp932 >> 8) & 0xFF);
          *out_ptr++ = static_cast<char>(cp932 & 0xFF);
        }
      }
    }

    if (!d_chars) { d_sizes[idx] = nbytes; }
  }
};

}  // namespace

std::unique_ptr<column> utf8_to_cp932(strings_column_view const& input,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }

  auto d_column = column_device_view::create(input.parent(), stream);

  // Allocate error flag on device
  rmm::device_scalar<bool> d_has_errors(false, stream);

  // Create the functor with error tracking
  utf8_to_cp932_fn fn{*d_column};
  fn.d_has_errors = d_has_errors.data();

  // Execute two-pass conversion
  auto [offsets_column, chars] = make_strings_children(fn, input.size(), stream, mr);

  // Check for conversion errors
  bool has_errors = d_has_errors.value(stream);
  CUDF_EXPECTS(!has_errors,
               "UTF-8 to CP932 conversion failed: input contains characters "
               "that cannot be represented in CP932 encoding");

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<column> utf8_to_cp932(strings_column_view const& input,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::utf8_to_cp932(input, stream, mr);
}

}  // namespace strings
}  // namespace cudf
