/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "binops_custom.cuh"

namespace cudf::binops::compiled {

// Specialize for NullEquals
template <>
struct ops_wrapper<ops::NullMin> {
  using BinaryOperator = ops::NullMin;
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  template <typename TypeCommon>
  __device__ void operator()(size_type i)
  {
    if constexpr (std::is_invocable_v<BinaryOperator, TypeCommon, TypeCommon>) {
      TypeCommon x      = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
      TypeCommon y      = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
      bool output_valid = false;
      auto result       = BinaryOperator{}.template operator()<TypeCommon, TypeCommon>(
        x, y, lhs.is_valid(i), rhs.is_valid(i), output_valid);
      type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
      if (out.nullable() && !output_valid) out.set_null(i);
    }
    (void)i;
  }
};

// Specialize for NullMin
template <>
struct ops2_wrapper<ops::NullMin> {
  using BinaryOperator = ops::NullMin;
  mutable_column_device_view& out;
  column_device_view const& lhs;
  column_device_view const& rhs;
  template <typename TypeLhs, typename TypeRhs>
  __device__ void operator()(size_type i)
  {
    if constexpr (!has_common_type_v<TypeLhs, TypeRhs> and
                  std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
      TypeLhs x         = lhs.element<TypeLhs>(i);
      TypeRhs y         = rhs.element<TypeRhs>(i);
      bool output_valid = false;
      auto result       = BinaryOperator{}.template operator()<TypeLhs, TypeRhs>(
        x, y, lhs.is_valid(i), rhs.is_valid(i), output_valid);
      type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
      if (out.nullable() && !output_valid) out.set_null(i);
    }
    (void)i;
  }
};

template void apply_binary_op<ops::NullMin>(mutable_column_device_view&,
                                            column_device_view const&,
                                            column_device_view const&,
                                            rmm::cuda_stream_view);
}  // namespace cudf::binops::compiled
