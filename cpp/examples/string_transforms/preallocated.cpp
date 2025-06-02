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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

std::unique_ptr<cudf::column> transform(cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  /// Convert a phone number to E.164 international phone format
  /// (https://en.wikipedia.org/wiki/E.164)
  auto udf = R"***(
__device__ void e164_format(void* scratch,
                            cudf::size_type row,
                            cudf::string_view* out,
                            cudf::string_view const country_code,
                            cudf::string_view const area_code,
                            cudf::string_view const phone_number,
                            [[maybe_unused]] int32_t scratch_size)
{
  auto const begin = static_cast<char*>(scratch) +
                     static_cast<ptrdiff_t>(row) * static_cast<ptrdiff_t>(scratch_size);
  auto const end = begin + scratch_size;
  auto it        = begin;

  auto push = [&](cudf::string_view str) {
    auto const size = str.size_bytes();

    if ((it + size) > end) { return; }

    memcpy(it, str.data(), size);
    it += size;
  };

  push(cudf::string_view{"+", 1});
  push(country_code);
  push(cudf::string_view{"-", 1});
  push(area_code);
  push(cudf::string_view{"-", 1});
  push(phone_number);

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - static_cast<char*>(begin))};
}
  )***";

  constexpr cudf::size_type maximum_size = 20;
  auto const num_rows                    = table.num_rows();

  rmm::device_uvector<char> scratch(maximum_size * num_rows, stream, mr);

  // a column with size 1 is considered a scalar
  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(maximum_size, true, stream, mr), 1, stream, mr);

  return cudf::transform({table.column(2), table.column(3), table.column(4), *size},
                         udf,
                         cudf::data_type{cudf::type_id::STRING},
                         false,
                         scratch.data(),
                         stream,
                         mr);
}
