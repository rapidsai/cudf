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

  auto country_code = table.column(2);
  auto area_code    = table.column(3);
  auto phone_number = table.column(4);

  auto country_symbol = cudf::make_column_from_scalar(
    cudf::string_scalar("+", true, stream, mr), country_code.size(), stream, mr);
  auto extension_symbol = cudf::make_column_from_scalar(
    cudf::string_scalar("-", true, stream, mr), country_code.size(), stream, mr);

  return cudf::strings::concatenate(
    cudf::table_view(
      {*country_symbol, *extension_symbol, area_code, *extension_symbol, phone_number}),
    cudf::string_scalar("", true, stream, mr),
    cudf::string_scalar("", false, stream, mr),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);
}
