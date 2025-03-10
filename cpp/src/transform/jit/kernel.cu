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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/jit/types.hpp>
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

template <typename T, int32_t index>
struct accessor {
  static_assert(cudf::is_rep_layout_compatible<T>(),
                "Accessor requires type to be memory-compatible with its representation");

  using type                     = T;
  static constexpr int32_t INDEX = index;

  static __device__ T& get(cudf::jit::column_device_view const* views, cudf::size_type row)
  {
    T* data = reinterpret_cast<T*>(views[INDEX].data);
    return data[row];
  }

  static __device__ void set(cudf::jit::column_device_view const* views,
                             cudf::size_type row,
                             T const& value)
  {
    T* data   = reinterpret_cast<T*>(views[INDEX].data);
    data[row] = value;
  }
};

template <typename Rep, int32_t index, numeric::Radix rad>
struct accessor<numeric::fixed_point<Rep, rad>, index> {
  using type                     = numeric::fixed_point<Rep, rad>;
  using rep                      = Rep;
  static constexpr int32_t INDEX = index;

  static __device__ type get(cudf::jit::column_device_view const* views, cudf::size_type row)
  {
    rep* data                 = reinterpret_cast<rep*>(views[INDEX].data);
    numeric::scale_type scale = static_cast<numeric::scale_type>(views[INDEX].type.scale());
    return type{numeric::scaled_integer<rep>{data[row], scale}};
  }

  static __device__ void set(cudf::jit::column_device_view const* views,
                             cudf::size_type row,
                             type const& value)
  {
    rep* data = reinterpret_cast<rep*>(views[INDEX].data);
    data[row] = value.value();
  }
};

template <typename accessor>
struct scalar {
  using type                     = typename accessor::type;
  static constexpr int32_t INDEX = accessor::INDEX;

  static __device__ decltype(auto) get(cudf::jit::column_device_view const* views,
                                       cudf::size_type row)
  {
    return accessor::get(views, 0);
  }

  static __device__ void set(cudf::jit::column_device_view const* views,
                             cudf::size_type row,
                             type const& value)
  {
    return accessor::set(views, 0, value);
  }
};

template <typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::size_type size, cudf::jit::column_device_view const* views)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size          = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start  = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride = block_size * gridDim.x;

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    GENERIC_TRANSFORM_OP(&Out::get(views, i), In::get(views, i)...);
  }
}

template <typename Out, typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::size_type size,
                                    cudf::jit::column_device_view const* views)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  auto const block_size                  = static_cast<thread_index_type>(blockDim.x);
  thread_index_type const start          = threadIdx.x + blockIdx.x * block_size;
  thread_index_type const stride         = block_size * gridDim.x;
  numeric::scale_type const output_scale = static_cast<numeric::scale_type>(views[0].type.scale());

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    typename Out::type result{numeric::scaled_integer<typename Out::rep>{0, output_scale}};
    GENERIC_TRANSFORM_OP(&result, In::get(views, i)...);
    Out::set(views, i, result);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
