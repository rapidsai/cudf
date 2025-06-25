

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

#pragma once
#include "jit/span.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>

#include <cstddef>

namespace cudf {
namespace jit {

template <typename T, int32_t Index>
struct column_accessor {
  using type                     = T;
  static constexpr int32_t index = Index;

  static __device__ decltype(auto) element(cudf::mutable_column_device_view_core const* outputs,
                                           cudf::size_type row)
  {
    return outputs[index].element<T>(row);
  }

  static __device__ decltype(auto) element(cudf::column_device_view_core const* inputs,
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

  static __device__ bool is_null(cudf::column_device_view_core const* inputs, cudf::size_type row)
  {
    return inputs[index].is_null(row);
  }

  static __device__ bool is_null(cudf::mutable_column_device_view_core const* inputs,
                                 cudf::size_type row)
  {
    return inputs[index].is_null(row);
  }
};

template <typename T, int32_t Index>
struct span_accessor {
  using type                     = T;
  static constexpr int32_t index = Index;

  static __device__ type& element(cudf::jit::device_optional_span<T> const* spans,
                                  cudf::size_type row)
  {
    return spans[index][row];
  }

  static __device__ void assign(cudf::jit::device_optional_span<T> const* outputs,
                                cudf::size_type row,
                                T value)
  {
    outputs[index][row] = value;
  }

  static __device__ bool is_null(cudf::jit::device_optional_span<T> const* inputs,
                                 cudf::size_type row)
  {
    return inputs[index].is_null(row);
  }
};

template <typename Accessor>
struct scalar_accessor {
  using type                     = typename Accessor::type;
  static constexpr int32_t index = Accessor::index;

  static __device__ decltype(auto) element(cudf::mutable_column_device_view_core const* outputs,
                                           cudf::size_type)
  {
    return Accessor::element(outputs, 0);
  }

  static __device__ decltype(auto) element(cudf::column_device_view_core const* inputs,
                                           cudf::size_type)
  {
    return Accessor::element(inputs, 0);
  }

  static __device__ void assign(cudf::mutable_column_device_view_core const* outputs,
                                cudf::size_type,
                                type value)
  {
    return Accessor::assign(outputs, 0, value);
  }

  static __device__ bool is_null(cudf::column_device_view_core const* inputs, cudf::size_type)
  {
    return Accessor::is_null(inputs, 0);
  }

  static __device__ bool is_null(cudf::mutable_column_device_view_core const* inputs,
                                 cudf::size_type)
  {
    return Accessor::is_null(inputs, 0);
  }
};

}  // namespace jit
}  // namespace cudf
