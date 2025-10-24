/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto udf = R"***(
__device__ void email_provider(cudf::string_view* out,
                               cudf::string_view const email,
                               cudf::string_view const alt)
{
  auto at_pos = email.find('@');

  if (at_pos == cudf::string_view::npos) {
    // malformed email, return alt
    *out = alt;
    return;
  }

  auto provider_begin = at_pos + 1;

  auto provider = email.substr(provider_begin, email.length() - provider_begin);

  // find the position of '.' in the provider
  auto dot_pos = provider.find('.');

  if (dot_pos == cudf::string_view::npos) {
    // malformed email, return alt
    *out = alt;
  } else {
    // return only the part before the dot
    *out = provider.substr(0, dot_pos);
  }
}
  )***";

  // a column with size 1 is considered a scalar
  auto alt = cudf::make_column_from_scalar(
    cudf::string_scalar(cudf::string_view{"(unknown)", 9}, true, stream, mr), 1, stream, mr);

  auto transformed = std::vector<int32_t>{1};
  auto emails      = table.column(1);

  auto providers = cudf::transform({emails, *alt},
                                   udf,
                                   cudf::data_type{cudf::type_id::STRING},
                                   false,
                                   std::nullopt,
                                   cudf::null_aware::NO,
                                   stream,
                                   mr);

  return {std::move(providers), std::move(transformed)};
}
