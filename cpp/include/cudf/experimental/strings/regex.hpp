/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/regex/flags.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace experimental {

std::unique_ptr<column> contains_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> matches_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> count_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> extract_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> extract_all_record_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> extract_single_jit(
  strings_column_view const& input,
  std::string_view pattern,
  size_type group,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> findall_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  strings::capture_groups captures  = strings::capture_groups::EXTRACT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> find_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> replace_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  string_scalar const& replacement           = string_scalar(""),
  std::optional<size_type> max_replace_count = std::nullopt,
  strings::regex_flags flags                 = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream               = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr          = cudf::get_current_device_resource_ref());

std::unique_ptr<column> replace_with_backrefs_jit(
  strings_column_view const& input,
  std::string_view pattern,
  std::string_view replacement,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> split_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  size_type maxsplit                = -1,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<table> rsplit_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  size_type maxsplit                = -1,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> split_record_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  size_type maxsplit                = -1,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<column> rsplit_record_re_jit(
  strings_column_view const& input,
  std::string_view pattern,
  size_type maxsplit                = -1,
  strings::regex_flags flags        = strings::regex_flags::DEFAULT,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace experimental
}  // namespace CUDF_EXPORT cudf
