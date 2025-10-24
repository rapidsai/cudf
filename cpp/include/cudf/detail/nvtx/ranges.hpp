/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvtx3/nvtx3.hpp>

#ifdef CUDF_ENABLE_NVTX_DEV_UTILS
#include <rmm/cuda_stream_view.hpp>
#endif

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

#ifdef CUDF_ENABLE_NVTX_DEV_UTILS
struct scoped_range_sync : scoped_range {
  rmm::cuda_stream_view stream;
  scoped_range_sync(char const* str, rmm::cuda_stream_view stream_)
    : scoped_range{str}, stream{stream_}
  {
  }
  ~scoped_range_sync() { stream.synchronize(); }
};
#endif

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

/**
 * @brief Conditionally compiled into `CUDF_FUNC_RANGE()` in debug build, and no-op otherwise.
 */
#ifndef NDEBUG
#define CUDF_FUNC_RANGE_DEBUG() CUDF_FUNC_RANGE()
#else  // no-op
#define CUDF_FUNC_RANGE_DEBUG()
#endif

/**
 * @brief Conditionally generate an instance of `cudf::scoped_range` in debug build, and no-op
 * otherwise.
 */
#ifndef NDEBUG
#define CUDF_SCOPED_RANGE_DEBUG(str) [[maybe_unused]] cudf::scoped_range __range{str};
#else  // no-op
#define CUDF_SCOPED_RANGE_DEBUG(str)
#endif

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `libcudf` domain with stream
 * synchronization at the end of the scope.
 *
 * This is an internal development utility, which should be compiled to nothing in production
 * builds. The NVTX range is generated and stream is synchronized only if the macro
 * `CUDF_ENABLE_NVTX_DEV_UTILS` is explicitly defined.
 *
 * Example:
 * ```
 * {
 *    CUDF_SCOPED_RANGE_SYNC_DEV("Search keys", stream);
 *    ...
 * }
 * ```
 */
#ifdef CUDF_ENABLE_NVTX_DEV_UTILS
#define CUDF_SCOPED_RANGE_SYNC_DEV(str, stream) \
  [[maybe_unused]] cudf::scoped_range_sync __range_sync{str, stream};
#else  // no-op
#define CUDF_SCOPED_RANGE_SYNC_DEV(str, stream)
#endif

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain from the lifetime
 * of a function, synchronizing the given stream at the end of the function.
 *
 * This is an internal development utility, which should be compiled to nothing in production
 * builds. The NVTX range is generated and stream is synchronized only if the macro
 * `CUDF_ENABLE_NVTX_DEV_UTILS` is explicitly defined.
 *
 * Similar to `CUDF_FUNC_RANGE`, this uses the name of the immediately enclosing function returned
 * by `__func__` to name the range.
 *
 * Example:
 * ```
 * void some_function(rmm::cuda_stream_view stream) {
 *    CUDF_FUNC_RANGE_SYNC_DEV(stream);
 *    ...
 * }
 * ```
 */
#ifdef CUDF_ENABLE_NVTX_DEV_UTILS
#define CUDF_FUNC_RANGE_SYNC_DEV(stream) \
  [[maybe_unused]] cudf::scoped_range_sync __range_sync{__func__, stream};
#else  // no-op
#define CUDF_FUNC_RANGE_SYNC_DEV(stream)
#endif
