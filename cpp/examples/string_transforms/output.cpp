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
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

std::unique_ptr<cudf::column> transform(cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto udf = R"***(
  __device__ void email_provider(cudf::string_view* out,
                                 cudf::string_view const email,
                                 cudf::string_view const alt)
  {
    auto pos = email.find('@');

    if (pos == cudf::string_view::npos) {
      *out = alt;
      return;
    }

    auto provider_begin = pos + 1;
    auto provider       = email.substr(provider_begin, email.length() - provider_begin);

    *out = provider;
  }
  )***";

  // a column with size 1 is considered a scalar
  auto alt = cudf::make_column_from_scalar(
    cudf::string_scalar(cudf::string_view{"(unknown)", 9}, true, stream, mr), 1, stream, mr);

  return cudf::transform({table.column(1), *alt},
                         udf,
                         cudf::data_type{cudf::type_id::STRING},
                         false,
                         std::nullopt,
                         stream,
                         mr);
}
