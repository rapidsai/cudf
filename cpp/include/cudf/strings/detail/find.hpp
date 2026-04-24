/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/find.hpp>

namespace cudf::strings::detail {

/**
 * @copydoc cudf::strings::contains
 */
std::unique_ptr<column> contains(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @copydoc cudf::strings::starts_with
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @copydoc cudf::strings::ends_with
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& input,
  string_scalar const& target,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace cudf::strings::detail
