

/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "jit/span.cuh"
#include "jit/accessors.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstddef>

// clang-format off
#include "filter/jit/operation-udf.hpp"
// clang-format on

namespace cudf {
namespace jit {

template <bool has_user_data, typename Out, typename... In>
CUDF_KERNEL void filter_kernel(cudf::jit::filter_output<Out> const* outputs,
                               cudf::column_device_view_core const* inputs,
                               void* user_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size           = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start   = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride  = block_size * gridDim.x;
  thread_index_type const size    = outputs[0].size();
  cudf::size_type num_not_applied = 0;

  for (auto i = start; i < size; i += stride) {
    auto const any_null = (false || ... || In::is_null(inputs, i));

    bool applies = false;

    if (!any_null) {
      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &applies, In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&applies, In::element(inputs, i)...);
      }
    }

    if (!applies) { num_not_applied++; }

    outputs[i] = applies ? i : -1;
  }

  atomicAdd(outputs->not_applied_count, num_not_applied);
}

}  // namespace jit
}  // namespace cudf
