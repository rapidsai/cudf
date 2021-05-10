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
namespace {
// Struct to launch only defined operations.
template <typename BinaryOperator>
struct ops_wrapper {
  template <typename T, typename... Args>
  __device__ enable_if_t<BinaryOperator::template is_supported<T>(), void> operator()(Args... args)
  {
    BinaryOperator{}.template operator()<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  __device__ enable_if_t<not BinaryOperator::template is_supported<T>(), void> operator()(
    Args... args)
  {
  }
};

// TODO merge these 2 structs somehow.
template <typename BinaryOperator>
struct ops2_wrapper {
  template <typename T1, typename T2, typename... Args>
  __device__ enable_if_t<BinaryOperator::template is_supported<T1, T2>(), void> operator()(
    Args... args)
  {
    BinaryOperator{}.template operator()<T1, T2>(std::forward<Args>(args)...);
  }

  template <typename T1, typename T2, typename... Args>
  __device__ enable_if_t<not BinaryOperator::template is_supported<T1, T2>(), void> operator()(
    Args... args)
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
      case binary_operator::ADD:                  dispatch_single_double<Add>(i); break;
      case binary_operator::SUB:                  dispatch_single_double<Sub>(i); break;
      case binary_operator::MUL:                  dispatch_single_double<Mul>(i); break;
      case binary_operator::DIV:                  dispatch_single_double<Div>(i); break;
      case binary_operator::TRUE_DIV:             dispatch_single_double<TrueDiv>(i); break;
      /*
      case binary_operator::FLOOR_DIV:            FloorDiv;
      case binary_operator::MOD:                  Mod;
      case binary_operator::PYMOD:                PyMod;
      case binary_operator::POW:                  Pow;
      case binary_operator::EQUAL:                Equal;
      case binary_operator::NOT_EQUAL:            NotEqual;
      case binary_operator::LESS:                 Less;
      case binary_operator::GREATER:              Greater;
      case binary_operator::LESS_EQUAL:           LessEqual;
      case binary_operator::GREATER_EQUAL:        GreaterEqual;
      case binary_operator::BITWISE_AND:          BitwiseAnd;
      case binary_operator::BITWISE_OR:           BitwiseOr;
      case binary_operator::BITWISE_XOR:          BitwiseXor;
      case binary_operator::LOGICAL_AND:          LogicalAnd;
      case binary_operator::LOGICAL_OR:           LogicalOr;
      case binary_operator::GENERIC_BINARY:       UserDefinedOp;
      case binary_operator::SHIFT_LEFT:           ShiftLeft;
      case binary_operator::SHIFT_RIGHT:          ShiftRight;
      case binary_operator::SHIFT_RIGHT_UNSIGNED: ShiftRightUnsigned;
      case binary_operator::LOG_BASE:             LogBase;
      case binary_operator::ATAN2:                ATan2;
      case binary_operator::PMOD:                 PMod;
      case binary_operator::NULL_EQUALS:          NullEquals;
      case binary_operator::NULL_MAX:             NullMax;
      case binary_operator::NULL_MIN:             NullMin; */
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

    // TODO move to utility.
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
