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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace binops {
namespace compiled {

void operator_dispatcher(mutable_column_view& out,
                         column_view const& lhs,
                         column_view const& rhs,
                         binary_operator op,
                         rmm::cuda_stream_view stream)
{
  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);

  // clang-format off
  switch (op) {
    case binary_operator::ADD:                  compiled_binary_op<ops::Add>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SUB:                  compiled_binary_op<ops::Sub>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::MUL:                  compiled_binary_op<ops::Mul>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::DIV:                  compiled_binary_op<ops::Div>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::TRUE_DIV:             compiled_binary_op<ops::TrueDiv>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::FLOOR_DIV:            compiled_binary_op<ops::FloorDiv>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::MOD:                  compiled_binary_op<ops::Mod>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::PYMOD:                compiled_binary_op<ops::PyMod>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::POW:                  compiled_binary_op<ops::Pow>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::EQUAL:                //compiled_binary_op<ops::Equal>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NOT_EQUAL:            //compiled_binary_op<ops::NotEqual>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NULL_EQUALS:          //compiled_binary_op<ops::NullEquals>(*outd, *lhsd, *rhsd, stream); break;
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_equality_op(*outd, *lhsd, *rhsd, op, stream); break;
    case binary_operator::LESS:                 //compiled_binary_op<ops::Less>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::GREATER:              //compiled_binary_op<ops::Greater>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LESS_EQUAL:           //compiled_binary_op<ops::LessEqual>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::GREATER_EQUAL:        //compiled_binary_op<ops::GreaterEqual>(*outd, *lhsd, *rhsd, stream); break;
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_comparison_op(*outd, *lhsd, *rhsd, op, stream); break;
    case binary_operator::BITWISE_AND:          compiled_binary_op<ops::BitwiseAnd>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::BITWISE_OR:           compiled_binary_op<ops::BitwiseOr>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::BITWISE_XOR:          compiled_binary_op<ops::BitwiseXor>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOGICAL_AND:          compiled_binary_op<ops::LogicalAnd>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOGICAL_OR:           compiled_binary_op<ops::LogicalOr>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_LEFT:           compiled_binary_op<ops::ShiftLeft>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_RIGHT:          compiled_binary_op<ops::ShiftRight>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_RIGHT_UNSIGNED: compiled_binary_op<ops::ShiftRightUnsigned>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOG_BASE:             compiled_binary_op<ops::LogBase>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::ATAN2:                compiled_binary_op<ops::ATan2>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::PMOD:                 compiled_binary_op<ops::PMod>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NULL_MAX:             compiled_binary_op<ops::NullMax>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NULL_MIN:             compiled_binary_op<ops::NullMin>(*outd, *lhsd, *rhsd, stream); break;
    /*
    case binary_operator::GENERIC_BINARY:       compiled_binary_op<ops::UserDefinedOp>(*outd, *lhsd, *rhsd, stream); break;
    */
    default:;
  }
  // clang-format on
}

// // TODO add boolean for scalars.
void binary_operation_compiled(mutable_column_view& out,
                               column_view const& lhs,
                               column_view const& rhs,
                               binary_operator op,
                               rmm::cuda_stream_view stream)
{
  operator_dispatcher(out, lhs, rhs, op, stream);
  // DONE vector_vector  //TODO vector_scalar, scalar_vector
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

  if (not binops::compiled::is_supported_operation(output_type, lhs.type(), rhs.type(), op))
    CUDF_FAIL("Unsupported operator for these types");

  // TODO check if scale conversion required?
  // if (is_fixed_point(lhs.type()) or is_fixed_point(rhs.type()))
  //  CUDF_FAIL("Not yet supported fixed_point");
  // return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if (lhs.is_empty() or rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
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
