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

#include "jit/accessors.cuh"
#include "jit/span.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>

// clang-format off
// This header is an inlined header that defines the GENERIC_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace filtering {
namespace jit {

template <bool has_user_data, typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::jit::device_optional_span<typename Out::type> const* outputs,
                        cudf::column_device_view_core const* inputs,
                        void* user_data)
{
  using index_type = typename Out::type;

  static constexpr index_type NOT_APPLIED = -1;

  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const output = outputs[0].to_span();
  auto const size   = output.size();

  for (auto i = start; i < size; i += stride) {
    auto const any_null = (false || ... || In::is_null(inputs, i));

    bool applies = false;

    if (!any_null) {
      if constexpr (has_user_data) {
        GENERIC_FILTER_OP(user_data, i, &applies, In::element(inputs, i)...);
      } else {
        GENERIC_FILTER_OP(&applies, In::element(inputs, i)...);
      }
    }

    output[i] = applies ? static_cast<index_type>(i) : NOT_APPLIED;
  }
}

}  // namespace jit
}  // namespace filtering
}  // namespace cudf
