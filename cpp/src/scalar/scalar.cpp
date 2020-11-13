/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <string>

namespace cudf {
std::string string_scalar::to_string(rmm::cuda_stream_view stream) const
{
  std::string result;
  result.resize(_data.size());
  CUDA_TRY(cudaMemcpyAsync(
    &result[0], _data.data(), _data.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return result;
}

}  // namespace cudf
