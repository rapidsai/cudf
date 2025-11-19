/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvtx3/nvtx3.hpp>

namespace cudf {
/**
 * @brief Tag type for libcudf's NVTX domain.
 */
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 *
 * Customizes an NVTX range with the given input.
 *
 * Example:
 * ```
 * void some_function(){
 *    cudf::scoped_range rng{"custom_name"}; // Customizes range name
 *    ...
 * }
 * ```
 */
using scoped_range = ::nvtx3::scoped_range_in<libcudf_domain>;

}  // namespace cudf

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define CUDF_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cudf::libcudf_domain)
