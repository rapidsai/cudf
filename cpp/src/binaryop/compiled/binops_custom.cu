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

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <binaryop/jit/operation.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {

namespace binops {
namespace compiled {
// Defined in util.cpp
data_type get_common_type(data_type out, data_type lhs, data_type rhs);

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

struct operator_dispatcher {
  //, OperatorType type)
  // (type == OperatorType::Direct ? operator_name : 'R' + operator_name);
  data_type common_data_type;
  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  binary_operator op;
  operator_dispatcher(data_type ct,
                      mutable_column_device_view ot,
                      column_device_view lt,
                      column_device_view rt,
                      binary_operator op)
    : common_data_type(ct), out(ot), lhs(lt), rhs(rt), op(op)
  {
  }

  template <class BinaryOperator>
  inline __device__ void dispatch_single_double(size_type i)
  {
    if (common_data_type == data_type{type_id::EMPTY}) {
      double_type_dispatcher(
        lhs.type(), rhs.type(), ops2_wrapper<BinaryOperator>{}, i, lhs, rhs, out);
    } else
      type_dispatcher(common_data_type, ops_wrapper<BinaryOperator>{}, i, lhs, rhs, out);
  }

  __device__ void operator()(size_type i)
  {
    // clang-format off
    switch (op) {
        // TODO One more level of indirection to allow double type dispatching for chrono types.
      case binary_operator::ADD:                  dispatch_single_double<ops::Add>(i); break;
      case binary_operator::SUB:                  dispatch_single_double<ops::Sub>(i); break;
      case binary_operator::MUL:                  dispatch_single_double<ops::Mul>(i); break;
      case binary_operator::DIV:                  dispatch_single_double<ops::Div>(i); break;
      case binary_operator::TRUE_DIV:             dispatch_single_double<ops::TrueDiv>(i); break;
      case binary_operator::FLOOR_DIV:            dispatch_single_double<ops::FloorDiv>(i); break;
      case binary_operator::MOD:                  dispatch_single_double<ops::Mod>(i); break;
      case binary_operator::PYMOD:                dispatch_single_double<ops::PyMod>(i); break;
      case binary_operator::POW:                  dispatch_single_double<ops::Pow>(i); break;
      case binary_operator::EQUAL:                dispatch_single_double<ops::Equal>(i); break;
      case binary_operator::NOT_EQUAL:            dispatch_single_double<ops::NotEqual>(i); break;
      case binary_operator::LESS:                 dispatch_single_double<ops::Less>(i); break;
      case binary_operator::GREATER:              dispatch_single_double<ops::Greater>(i); break;
      case binary_operator::LESS_EQUAL:           dispatch_single_double<ops::LessEqual>(i); break;
      case binary_operator::GREATER_EQUAL:        dispatch_single_double<ops::GreaterEqual>(i); break;
      case binary_operator::BITWISE_AND:          dispatch_single_double<ops::BitwiseAnd>(i); break;
      case binary_operator::BITWISE_OR:           dispatch_single_double<ops::BitwiseOr>(i); break;
      case binary_operator::BITWISE_XOR:          dispatch_single_double<ops::BitwiseXor>(i); break;
      case binary_operator::LOGICAL_AND:          dispatch_single_double<ops::LogicalAnd>(i); break;
      case binary_operator::LOGICAL_OR:           dispatch_single_double<ops::LogicalOr>(i); break;
      case binary_operator::SHIFT_LEFT:           dispatch_single_double<ops::ShiftLeft>(i); break;
      case binary_operator::SHIFT_RIGHT:          dispatch_single_double<ops::ShiftRight>(i); break;
      case binary_operator::SHIFT_RIGHT_UNSIGNED: dispatch_single_double<ops::ShiftRightUnsigned>(i); break;
      case binary_operator::LOG_BASE:             dispatch_single_double<ops::LogBase>(i); break;
      case binary_operator::ATAN2:                dispatch_single_double<ops::ATan2>(i); break;
      case binary_operator::PMOD:                 dispatch_single_double<ops::PMod>(i); break;
      /*
      case binary_operator::NULL_EQUALS:          dispatch_single_double<ops::NullEquals>(i); break;
      case binary_operator::NULL_MAX:             dispatch_single_double<ops::NullMax>(i); break;
      case binary_operator::NULL_MIN:             dispatch_single_double<ops::NullMin>(i); break;
      case binary_operator::GENERIC_BINARY:       dispatch_single_double<ops::UserDefinedOp>(i); break;
      */
      default:                                    ;
    }
    // clang-format on
  }
};

}  // namespace

void binary_operation_compiled(mutable_column_view& out,
                               column_view const& lhs,
                               column_view const& rhs,
                               binary_operator op,
                               rmm::cuda_stream_view stream)
{
  if (is_null_dependent(op)) {
    CUDF_FAIL("Unsupported yet");
    // cudf::binops::jit::kernel_v_v_with_validity
  } else {
    // Create binop functor instance
    auto lhsd = column_device_view::create(lhs, stream);
    auto rhsd = column_device_view::create(rhs, stream);
    auto outd = mutable_column_device_view::create(out, stream);
    // auto binop_func = device_dispatch_functor<cudf::binops::jit::Add2>{*lhsd, *rhsd, *outd};

    auto common_dtype = get_common_type(out.type(), lhs.type(), rhs.type());
    if (not(op == binary_operator::ADD or op == binary_operator::SUB or
            op == binary_operator::MUL or op == binary_operator::DIV or
            op == binary_operator::TRUE_DIV))
      CUDF_FAIL("Unsupported operator");
    // Execute it on every element
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(out.size()),
                     operator_dispatcher{common_dtype, *outd, *lhsd, *rhsd, op});
    //"cudf::binops::jit::kernel_v_v")  //
  }
}
}  // namespace compiled
}  // namespace binops

namespace detail {

std::unique_ptr<column> make_fixed_width_column_for_output(column_view const& lhs,
                                                           column_view const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr);

std::unique_ptr<column> binary_operation_compiled(column_view const& lhs,
                                                  column_view const& rhs,
                                                  binary_operator op,
                                                  data_type output_type,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes don't match");

  if (lhs.type().id() == type_id::STRING and rhs.type().id() == type_id::STRING)
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, stream, mr);

  if (is_fixed_point(lhs.type()) or is_fixed_point(rhs.type()))
    CUDF_FAIL("Not yet supported fixed_point");
  // return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type), "Invalid/Unsupported output datatype");
  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if (lhs.is_empty() or rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  // CUDF_FAIL("Not yet supported fixed_width");
  binops::compiled::binary_operation_compiled(out_view, lhs, rhs, op, stream);
  return out;
}
}  // namespace detail

std::unique_ptr<column> binary_operation_compiled(column_view const& lhs,
                                                  column_view const& rhs,
                                                  binary_operator op,
                                                  data_type output_type,
                                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation_compiled(lhs, rhs, op, output_type, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
