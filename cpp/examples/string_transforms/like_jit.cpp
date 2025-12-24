/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace {
std::unique_ptr<cudf::column> like(cudf::strings_column_view const& input,
                                   std::string const& target1,
                                   std::string const& target2,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  auto const ss1  = cudf::string_scalar(target1, true, stream);
  auto const ss2  = cudf::string_scalar(target2, true, stream);
  auto col_first  = cudf::make_column_from_scalar(ss1, 1, stream, mr);
  auto col_second = cudf::make_column_from_scalar(ss2, 1, stream, mr);

  auto udf = R"***(
 __device__ void like_kernel(bool* out,
                             cudf::string_view const d_str,
                             cudf::string_view const d_first,
                             cudf::string_view const d_second)
{
  auto found = d_str.find(d_first);
  if (found != d_str.npos) { found = d_str.find(d_second, found + d_first.length()); }
  *out = found != d_str.npos;
}
   )***";

  auto results = cudf::transform({input.parent(), col_first->view(), col_second->view()},
                                 udf,
                                 cudf::data_type{cudf::type_id::BOOL8},
                                 false,
                                 std::nullopt,
                                 cudf::null_aware::NO,
                                 stream,
                                 mr);
  return results;
}
}  // namespace

std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table)
{
  auto stream = rmm::cuda_stream_default;
  auto mr     = cudf::get_current_device_resource_ref();

  auto transformed = std::vector<int32_t>{0};
  auto comments    = table.column(0);

  // std::string pattern = "%%special%%requests%%";
  std::string target1 = "special";
  std::string target2 = "requests";

  auto sv     = cudf::strings_column_view(comments);
  auto result = like(sv, target1, target2, stream, mr);

  return std::make_tuple(std::move(result), transformed);
}
