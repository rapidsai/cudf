/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup default_stream
 * @{
 * @file
 */

/**
 * @brief Get the current default stream
 *
 * @return The current default stream.
 */
rmm::cuda_stream_view const get_default_stream();

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled();

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
