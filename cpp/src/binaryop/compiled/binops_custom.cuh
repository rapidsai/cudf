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
#pragma once

#include "binary_ops.hpp"
#include "operation.cuh"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {

namespace binops {
namespace compiled {

namespace {
// Struct to launch only defined operations.
template <typename BinaryOperator>
struct ops_wrapper {
  template <typename TypeCommon,
            std::enable_if_t<is_op_supported<TypeCommon, TypeCommon, BinaryOperator>()>* = nullptr>
  __device__ void operator()(size_type i,
                             column_device_view const lhs,
                             column_device_view const rhs,
                             mutable_column_device_view const out)
  {
    TypeCommon x = type_dispatcher(lhs.type(), type_casted_accessor<TypeCommon>{}, i, lhs);
    TypeCommon y = type_dispatcher(rhs.type(), type_casted_accessor<TypeCommon>{}, i, rhs);
    auto result  = BinaryOperator{}.template operator()<TypeCommon, TypeCommon>(x, y);
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }

  template <
    typename TypeCommon,
    typename... Args,
    std::enable_if_t<not is_op_supported<TypeCommon, TypeCommon, BinaryOperator>()>* = nullptr>
  __device__ void operator()(Args... args)
  {
  }
};

// TODO merge these 2 structs somehow.
template <typename BinaryOperator>
struct ops2_wrapper {
  template <typename TypeLhs,
            typename TypeRhs,
            std::enable_if_t<!has_common_type_v<TypeLhs, TypeRhs> and
                             is_op_supported<TypeLhs, TypeRhs, BinaryOperator>()>* = nullptr>
  __device__ void operator()(size_type i,
                             column_device_view const lhs,
                             column_device_view const rhs,
                             mutable_column_device_view const out)
  {
    TypeLhs x   = lhs.element<TypeLhs>(i);
    TypeRhs y   = rhs.element<TypeRhs>(i);
    auto result = BinaryOperator{}.template operator()<TypeLhs, TypeRhs>(x, y);
    //(void)result;
    type_dispatcher(out.type(), typed_casted_writer<decltype(result)>{}, i, out, result);
  }

  template <typename TypeLhs,
            typename TypeRhs,
            typename... Args,
            std::enable_if_t<has_common_type_v<TypeLhs, TypeRhs> or
                             not is_op_supported<TypeLhs, TypeRhs, BinaryOperator>()>* = nullptr>
  __device__ void operator()(Args... args)
  {
  }
};

template <class BinaryOperator>
struct device_type_dispatcher {
  //, OperatorType type)
  // (type == OperatorType::Direct ? operator_name : 'R' + operator_name);
  data_type common_data_type;
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  device_type_dispatcher(data_type ct,
                         mutable_column_device_view ot,
                         column_device_view lt,
                         column_device_view rt)
    : common_data_type(ct), out(ot), lhs(lt), rhs(rt)
  {
  }

  __device__ void operator()(size_type i)
  {
    if (common_data_type == data_type{type_id::EMPTY}) {
      double_type_dispatcher(
        lhs.type(), rhs.type(), ops2_wrapper<BinaryOperator>{}, i, lhs, rhs, out);
    } else {
      type_dispatcher(common_data_type, ops_wrapper<BinaryOperator>{}, i, lhs, rhs, out);
    }
  }
};
}  // namespace

template <class BinaryOperator>
void dispatch_single_double(mutable_column_device_view& outd,
                            column_device_view const& lhsd,
                            column_device_view const& rhsd,
                            rmm::cuda_stream_view stream)
{
  auto common_dtype = get_common_type(outd.type(), lhsd.type(), rhsd.type());

  // Create binop functor instance
  auto binop_func = device_type_dispatcher<BinaryOperator>{common_dtype, outd, lhsd, rhsd};
  // Execute it on every element
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(outd.size()),
                   binop_func);
  //"cudf::binops::jit::kernel_v_v")  //
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
