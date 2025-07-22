/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "common.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto udf = R"***(
 __device__ void checksum(uint16_t* out,
                          cudf::string_view const name,
                          cudf::string_view const email)
 {
   auto fletcher16 = [](cudf::string_view str) -> uint16_t {
     uint16_t sum1 = 0;
     uint16_t sum2 = 0;
     for (cudf::size_type i = 0; i < str.size_bytes(); ++i) {
       sum1 = (sum1 + str.data()[i]) % 255;
       sum2 = (sum2 + sum1) % 255;
     }
     return (sum2 << 8) | sum1;
   };
   *out = fletcher16(name) ^ fletcher16(email);
 }
   )***";

  auto transformed = std::vector<int32_t>{0, 1};
  auto name        = table.column(0);
  auto email       = table.column(1);

  auto result = cudf::transform(
    {name, email}, udf, cudf::data_type{cudf::type_id::UINT16}, false, std::nullopt, stream, mr);

  return std::make_tuple(std::move(result), transformed);
}
