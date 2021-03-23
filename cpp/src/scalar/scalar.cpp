/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/strings/string_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <string>

namespace cudf {

string_scalar::string_scalar(rmm::device_scalar<value_type>& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : string_scalar(data.value(stream), is_valid, stream, mr)
{
}

string_scalar::string_scalar(value_type const& source,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
  : scalar(data_type(type_id::STRING), is_valid),
    _data(source.data(), source.size_bytes(), stream, mr)
{
}

string_scalar::value_type string_scalar::value(rmm::cuda_stream_view stream) const
{
  return value_type{data(), size()};
}

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
