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
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
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

  auto area_iter       = area_code.data();
  auto const area_end  = area_iter + area_code.size_bytes();
  auto phone_iter      = phone_number.data();
  auto const phone_end = phone_iter + phone_number.size_bytes();

  // skip leading zeros in area code and push non-dash digits
  while (area_iter != area_end && *area_iter == '0') {
    area_iter++;
  }

  while (area_iter != area_end) {
    if (*area_iter != '-') { push(cudf::string_view{area_iter, 1}); }
    area_iter++;
  }

  // push non-dash digits from phone number
  while (phone_iter != phone_end) {
    if (*phone_iter != '-') { push(cudf::string_view{phone_iter, 1}); }
    phone_iter++;
  }

  *out = cudf::string_view{begin, static_cast<cudf::size_type>(it - static_cast<char*>(begin))};
}
  )***";

  constexpr cudf::size_type maximum_size = 20;
  auto const num_rows                    = table.num_rows();

  rmm::device_uvector<char> scratch(maximum_size * num_rows, stream, mr);

  // a column with size 1 is considered a scalar
  auto size = cudf::make_column_from_scalar(
    cudf::numeric_scalar<int32_t>(maximum_size, true, stream, mr), 1, stream, mr);

  auto formatted = cudf::transform({table.column(2), table.column(3), table.column(4), *size},
                                   udf,
                                   cudf::data_type{cudf::type_id::STRING},
                                   false,
                                   scratch.data(),
                                   stream,
                                   mr);

  return formatted;
}
