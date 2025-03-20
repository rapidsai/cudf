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

#include <cudf/jit/types.cuh>
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
struct accessor {
  using type                     = T;
  static constexpr int32_t index = Index;

  static __device__ decltype(auto) element(cudf::jit::mutable_column_device_view const* views,
                                           cudf::size_type row)
  {
    return views[index].element<T>(row);
  }

  static __device__ decltype(auto) element(cudf::jit::column_device_view const* views,
                                           cudf::size_type row)
  {
    return views[index].element<T>(row);
  }

  static __device__ void assign(cudf::jit::mutable_column_device_view const* views,
                                cudf::size_type row,
                                T value)
  {
    views[index].assign<T>(row, value);
  }
};

template <typename Accessor>
struct scalar {
  using type                     = typename Accessor::type;
  static constexpr int32_t index = Accessor::index;

  static __device__ decltype(auto) element(cudf::jit::mutable_column_device_view const* views,
                                           cudf::size_type row)
  {
    return Accessor::element(views, 0);
  }

  static __device__ decltype(auto) element(cudf::jit::column_device_view const* views,
                                           cudf::size_type row)
  {
    return Accessor::element(views, 0);
  }

  static __device__ void assign(cudf::jit::mutable_column_device_view const* views,
                                cudf::size_type row,
                                type value)
  {
    return Accessor::assign(views, 0, value);
  }
};

template <typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::jit::mutable_column_device_view const* output,
                        cudf::jit::column_device_view const* inputs)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;
  thread_index_type const size   = output->size();

  for (auto i = start; i < size; i += stride) {
    GENERIC_TRANSFORM_OP(&Out::element(output, i), In::element(inputs, i)...);
  }
}

template <typename Out, typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::jit::mutable_column_device_view const* output,
                                    cudf::jit::column_device_view const* inputs)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;
  thread_index_type const size   = output->size();

  numeric::scale_type const output_scale = static_cast<numeric::scale_type>(output->type().scale());

  for (auto i = start; i < size; i += stride) {
    typename Out::type result{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}};
    GENERIC_TRANSFORM_OP(&result, In::element(inputs, i)...);
    Out::assign(output, i, result);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
