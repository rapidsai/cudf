/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
//
strings_column_view::strings_column_view(column_view strings_column) : column_view(strings_column)
{
  CUDF_EXPECTS(type().id() == type_id::STRING, "strings_column_view only supports strings");
}

strings_column_view::strings_column_view(column_view strings_column, rmm::cuda_stream_view stream)
  : column_view(strings_column)
{
  CUDF_EXPECTS(type().id() == type_id::STRING, "strings_column_view only supports strings");
  auto const [min_len, max_len] =
    cudf::strings::detail::compute_min_max_string_lengths(*this, stream);
  _min_length = min_len;
  _max_length = max_len;
}

column_view strings_column_view::parent() const { return static_cast<column_view>(*this); }

column_view strings_column_view::offsets() const
{
  CUDF_EXPECTS(num_children() > 0, "strings column has no children");
  return child(offsets_column_index);
}

int64_t strings_column_view::chars_size(rmm::cuda_stream_view stream) const
{
  if (size() == 0) { return 0L; }
  return cudf::strings::detail::get_offset_value(offsets(), offsets().size() - 1, stream);
}

strings_column_view::chars_iterator strings_column_view::chars_begin(
  rmm::cuda_stream_view) const noexcept
{
  return head<char>();
}

strings_column_view::chars_iterator strings_column_view::chars_end(
  rmm::cuda_stream_view stream) const
{
  return chars_begin(stream) + chars_size(stream);
}

int64_t strings_column_view::minimum() const
{
  CUDF_EXPECTS(_min_length >= 0,
               "minimum() requires construction via strings_column_view(column_view, stream)");
  return _min_length;
}

int64_t strings_column_view::maximum() const
{
  CUDF_EXPECTS(_max_length >= 0,
               "maximum() requires construction via strings_column_view(column_view, stream)");
  return _max_length;
}

}  // namespace cudf
