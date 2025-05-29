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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

std::unique_ptr<cudf::column> transform(cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto regex = cudf::strings::regex_program::create(".*@(.*)");

  auto alt = cudf::string_scalar(cudf::string_view{"(unknown)", 9}, true, stream, mr);

  auto emails = cudf::strings::extract_single(table.column(1), *regex, 0);

  auto emails_filled = cudf::replace_nulls(*emails, alt, stream, mr);

  return emails_filled;
}
