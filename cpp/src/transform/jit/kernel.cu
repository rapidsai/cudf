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

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/jit/raw_span.cuh>
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
#include "transform/jit/operation-udf.hpp"
// clang-format on

namespace cudf {
namespace transformation {
namespace jit {

template <typename T, int32_t Index>
struct column_accessor {
  using type                     = T;
  static constexpr int32_t index = Index;

  static __device__ decltype(auto) element(cudf::mutable_column_device_view_core const* outputs,
                                           [[maybe_unused]] void* const* user_data,
                                           cudf::size_type row)
  {
    return outputs[index].element<T>(row);
  }

  static __device__ decltype(auto) element(cudf::column_device_view_core const* inputs,
                                           [[maybe_unused]] void* const* user_data,
                                           cudf::size_type row)
  {
    return inputs[index].element<T>(row);
  }

  static __device__ void assign(cudf::mutable_column_device_view_core const* outputs,
                                cudf::size_type row,
                                T value)
  {
    outputs[index].assign<T>(row, value);
  }
};

template <typename T, int32_t Index>
struct span_accessor {
  using type                     = T;
  static constexpr int32_t index = Index;

  static __device__ type& element(raw_span<T> const* spans,
                                  [[maybe_unused]] void* const* user_data,
                                  cudf::size_type row)
  {
    return spans[index][row];
  }

  static __device__ void assign(raw_span<T> const* outputs, cudf::size_type row, T value)
  {
    outputs[index][row] = value;
  }
};

template <int32_t Index>
struct user_data_accessor {
  using type                     = void*;
  static constexpr int32_t index = Index;

  static __device__ void* element([[maybe_unused]] cudf::column_device_view_core const* inputs,
                                  void* const* user_data,
                                  [[maybe_unused]] cudf::size_type row)
  {
    return user_data[index];
  }
};

template <typename Accessor>
struct scalar {
  using type                     = typename Accessor::type;
  static constexpr int32_t index = Accessor::index;

  static __device__ decltype(auto) element(cudf::mutable_column_device_view_core const* outputs,
                                           void* const* user_data,
                                           cudf::size_type row)
  {
    return Accessor::element(outputs, user_data, 0);
  }

  static __device__ decltype(auto) element(cudf::column_device_view_core const* inputs,
                                           void* const* user_data,
                                           cudf::size_type row)
  {
    return Accessor::element(inputs, user_data, 0);
  }

  static __device__ void assign(cudf::mutable_column_device_view_core const* outputs,
                                cudf::size_type row,
                                type value)
  {
    return Accessor::assign(outputs, 0, value);
  }
};

template <typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::mutable_column_device_view_core const* outputs,
                        cudf::column_device_view_core const* inputs,
                        void* const* user_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;
  thread_index_type const size   = outputs[0].size();

  for (auto i = start; i < size; i += stride) {
    GENERIC_TRANSFORM_OP(&Out::element(outputs, user_data, i),
                         In::element(inputs, user_data, i)...);
  }
}

template <typename Out, typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::mutable_column_device_view_core const* outputs,
                                    cudf::column_device_view_core const* inputs,
                                    void* const* user_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;
  thread_index_type const size   = outputs[0].size();
  auto const output_scale        = static_cast<numeric::scale_type>(outputs[0].type().scale());

  for (auto i = start; i < size; i += stride) {
    typename Out::type result{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}};
    GENERIC_TRANSFORM_OP(&result, In::element(inputs, user_data, i)...);
    Out::assign(outputs, i, result);
  }
}

template <typename Out, typename... In>
CUDF_KERNEL void span_kernel(raw_span<typename Out::type> const* outputs,
                             cudf::column_device_view_core const* inputs,
                             void* const* user_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;
  thread_index_type const size   = outputs[0].size();

  for (auto i = start; i < size; i += stride) {
    GENERIC_TRANSFORM_OP(&Out::element(outputs, user_data, i),
                         In::element(inputs, user_data, i)...);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
