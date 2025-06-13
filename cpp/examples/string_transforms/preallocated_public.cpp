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

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto country_code   = table.column(2);
  auto area_code      = table.column(3);
  auto phone_number   = table.column(4);
  auto transformed    = std::vector<int32_t>{2, 3, 4};
  auto const num_rows = country_code.size();

  // remove all leading zeros from country code and any dashes
  auto stripped_country_code = cudf::strings::strip(country_code,
                                                    cudf::strings::side_type::LEFT,
                                                    cudf::string_scalar("0", true, stream, mr),
                                                    stream,
                                                    mr);

  stripped_country_code = cudf::strings::replace(cudf::strings_column_view(*stripped_country_code),
                                                 cudf::string_scalar("-", true, stream, mr),
                                                 cudf::string_scalar("", true, stream, mr),
                                                 -1,  // maxrepl: replace all occurrences
                                                 stream,
                                                 mr);

  // remove any dashes from phone number
  auto stripped_phone_number = cudf::strings::replace(cudf::strings_column_view(phone_number),
                                                      cudf::string_scalar("-", true, stream, mr),
                                                      cudf::string_scalar("", true, stream, mr),
                                                      -1,  // maxrepl: replace all occurrences
                                                      stream,
                                                      mr);

  auto country_symbol =
    cudf::make_column_from_scalar(cudf::string_scalar("+", true, stream, mr), num_rows, stream, mr);

  std::vector<cudf::column_view> concat_cols{
    *country_symbol, *stripped_country_code, area_code, *stripped_phone_number};

  auto formatted = cudf::strings::concatenate(cudf::table_view(concat_cols),
                                              cudf::string_scalar("", true, stream, mr),
                                              cudf::string_scalar("", true, stream, mr),
                                              cudf::strings::separator_on_nulls::YES,
                                              stream,
                                              mr);

  return std::make_tuple(std::move(formatted), transformed);
}
