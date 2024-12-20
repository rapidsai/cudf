/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
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
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/type_traits>

// clang-format off
#include "binaryop/jit/operation-udf.hpp"
// clang-format on

namespace cudf {
namespace binops {
namespace jit {

struct UserDefinedOp {
  template <typename TypeOut, typename TypeLhs, typename TypeRhs>
  static TypeOut operate(TypeLhs x, TypeRhs y)
  {
    TypeOut output;
    using TypeCommon = typename cuda::std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
    GENERIC_BINARY_OP(&output, static_cast<TypeCommon>(x), static_cast<TypeCommon>(y));
    return output;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
CUDF_KERNEL void kernel_v_v(cudf::size_type size,
                            TypeOut* out_data,
                            TypeLhs* lhs_data,
                            TypeRhs* rhs_data)
{
  auto const start = threadIdx.x + static_cast<cudf::thread_index_type>(blockIdx.x) * blockDim.x;
  auto const step  = static_cast<cudf::thread_index_type>(blockDim.x) * gridDim.x;

  for (auto i = start; i < size; i += step) {
    out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[i]);
  }
}

template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
CUDF_KERNEL void kernel_v_v_with_validity(cudf::size_type size,
                                          TypeOut* out_data,
                                          TypeLhs* lhs_data,
                                          TypeRhs* rhs_data,
                                          cudf::bitmask_type* output_mask,
                                          cudf::bitmask_type const* lhs_mask,
                                          cudf::size_type lhs_offset,
                                          cudf::bitmask_type const* rhs_mask,
                                          cudf::size_type rhs_offset)
{
  auto const start = threadIdx.x + static_cast<cudf::thread_index_type>(blockIdx.x) * blockDim.x;
  auto const step  = static_cast<cudf::thread_index_type>(blockDim.x) * gridDim.x;

  for (auto i = start; i < size; i += step) {
    bool output_valid = false;
    out_data[i]       = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(
      lhs_data[i],
      rhs_data[i],
      lhs_mask ? cudf::bit_is_set(lhs_mask, lhs_offset + i) : true,
      rhs_mask ? cudf::bit_is_set(rhs_mask, rhs_offset + i) : true,
      output_valid);
    if (output_mask && !output_valid) cudf::clear_bit(output_mask, i);
  }
}

}  // namespace jit
}  // namespace binops
}  // namespace cudf
