/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nvtx3/nvtx3.hpp>

#ifdef CUDF_ENABLE_NVTX_DEBUG_RANGE
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

#ifdef CUDF_ENABLE_NVTX_DEBUG_RANGE
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
 * @brief Convenience macro for generating an NVTX scoped range in the `libcudf` domain with stream
 * synchronization at the end of the scope.
 *
 * This is an internal development utility, which should be compiled to nothing in production
 * builds. The NVTX range is generated and stream is synchronized only if the macro
 * `CUDF_ENABLE_NVTX_DEBUG_RANGE` is explicitly defined.
 *
 * Example:
 * ```
 * {
 *    CUDF_SCOPED_RANGE_DEBUG("Search keys", stream);
 *    ...
 * }
 * ```
 */
#ifdef CUDF_ENABLE_NVTX_DEBUG_RANGE
#define CUDF_SCOPED_RANGE_DEBUG(str, stream) \
  [[maybe_unused]] cudf::scoped_range_sync __range_sync{str, stream};
#else  // no-op
#define CUDF_SCOPED_RANGE_DEBUG(str, stream)
#endif

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain from the lifetime
 * of a function, synchronizing the given stream at the end of the function.
 *
 * This is an internal development utility, which should be compiled to nothing in production
 * builds. The NVTX range is generated and stream is synchronized only if the macro
 * `CUDF_ENABLE_NVTX_DEBUG_RANGE` is explicitly defined.
 *
 * Similar to `CUDF_FUNC_RANGE`, this uses the name of the immediately enclosing function returned
 * by `__func__` to name the range.
 *
 * Example:
 * ```
 * void some_function(rmm::cuda_stream_view stream){
 *    CUDF_FUNC_RANGE_DEBUG(stream);
 *    ...
 * }
 * ```
 */
#ifdef CUDF_ENABLE_NVTX_DEBUG_RANGE
#define CUDF_FUNC_RANGE_DEBUG(stream) \
  [[maybe_unused]] cudf::scoped_range_sync __range_sync{__func__, stream};
#else  // no-op
#define CUDF_FUNC_RANGE_DEBUG(stream)
#endif
