/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/timezone.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::make_timezone_transition_table(std::optional<std::string_view>, std::string_view,
 * rmm::cuda_stream_view, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<table> make_timezone_transition_table(
  std::optional<std::string_view> tzif_dir,
  std::string_view timezone_name,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
