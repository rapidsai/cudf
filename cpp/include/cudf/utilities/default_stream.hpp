/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>

#include <cudf/utilities/error.hpp>

namespace cudf {

namespace detail {
#if __cpp_inline_variables
inline cudaStream_t default_stream;
#else
extern cudaStream_t default_stream;
#endif
}  // namespace detail

/**
 * @brief Set the default CUDA stream. This can only be done once.
 *
 * @param stream The stream to set as the default. Only `cudaStreamLegacy` or `cudaStreamPerThread`
 * is allowed.
 */
void set_default_stream(cudaStream_t stream)
{
  CUDF_EXPECTS(detail::default_stream == cudaStreamDefault, "Default stream can only be set once");
  CUDF_EXPECTS(stream == cudaStreamLegacy || stream == cudaStreamPerThread,
               "Default stream can only be set to legacy or per-thread");
  detail::default_stream = stream;
}

/**
 * @brief Get the default CUDA stream.
 *
 * @returns cudaStream_t The default CUDA stream.
 */
cudaStream_t get_default_stream() { return detail::default_stream; }

}  // namespace cudf
