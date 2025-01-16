/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

/**
 * @brief Redacts each name per the corresponding visibility entry
 *
 * This implementation uses libcudf APIs to create the output result.
 *
 * @param names Column of names
 * @param visibilities Column of visibilities
 * @return Redacted column of names
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities)
{
  auto const visible   = cudf::string_scalar(std::string("public"));
  auto const redaction = cudf::string_scalar(std::string("X X"));

  nvtxRangePushA("redact_strings");

  auto const allowed      = cudf::strings::contains(visibilities, visible);
  auto const redacted     = cudf::copy_if_else(names, redaction, allowed->view());
  auto const first_last   = cudf::strings::split(redacted->view());
  auto const first        = first_last->view().column(0);
  auto const last         = first_last->view().column(1);
  auto const last_initial = cudf::strings::slice_strings(last, 0, 1);

  auto const last_initial_first = cudf::table_view({last_initial->view(), first});

  auto result = cudf::strings::concatenate(last_initial_first, std::string_view(" "));

  cudaStreamSynchronize(0);

  nvtxRangePop();
  return result;
}
