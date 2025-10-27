/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvtx3/nvtx3.hpp>

namespace cudf::benchmark {
/**
 * @brief Tag type for the NVTX domain
 */
struct benchmark_domain {
  static constexpr char const* name{"benchmarks"};  ///< Name of the domain
};

}  // namespace cudf::benchmark

/**
 * @brief Convenience macro for generating an NVTX range in the `benchmarks` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_BENCHMARK_RANGE();
 *    ...
 * }
 * ```
 */
#define CUDF_BENCHMARK_RANGE() NVTX3_FUNC_RANGE_IN(cudf::benchmark::benchmark_domain)
