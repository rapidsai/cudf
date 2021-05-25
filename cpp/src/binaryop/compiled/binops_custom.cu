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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <binaryop/jit/operation.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf {

namespace binops {
namespace compiled {
// Defined in util.cpp
data_type get_common_type(data_type out, data_type lhs, data_type rhs);
bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op);

// extern templates
// TODO add boolean for scalars.
// clang-format off
extern template void dispatch_single_double<ops::Add>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::Sub>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::Mul>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::Div>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::TrueDiv>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::FloorDiv>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::Mod>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::PyMod>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::Pow>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::Equal>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::NotEqual>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::Less>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::Greater>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::LessEqual>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::GreaterEqual>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::BitwiseAnd>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::BitwiseOr>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::BitwiseXor>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::LogicalAnd>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::LogicalOr>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::ShiftLeft>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::ShiftRight>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::ShiftRightUnsigned>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::LogBase>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::ATan2>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
extern template void dispatch_single_double<ops::PMod>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::NullEquals>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::NullMax>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::NullMin>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// extern template void dispatch_single_double<ops::UserDefinedOp>(mutable_column_device_view&, column_device_view const&, column_device_view const&, rmm::cuda_stream_view);
// clang-format on
void dispatch_comparison_op(mutable_column_device_view& outd,
                            column_device_view const& lhsd,
                            column_device_view const& rhsd,
                            binary_operator op,
                            rmm::cuda_stream_view stream);
void dispatch_equality_op(mutable_column_device_view& outd,
                          column_device_view const& lhsd,
                          column_device_view const& rhsd,
                          binary_operator op,
                          rmm::cuda_stream_view stream);

void operator_dispatcher(mutable_column_view& out,
                         column_view const& lhs,
                         column_view const& rhs,
                         binary_operator op,
                         rmm::cuda_stream_view stream)
{
  if (not is_supported_operation(out.type(), lhs.type(), rhs.type(), op))
    CUDF_FAIL("Unsupported operator for these types");

  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);

  // clang-format off
  switch (op) {
      // TODO One more level of indirection to allow double type dispatching for chrono types.
    case binary_operator::ADD:                  dispatch_single_double<ops::Add>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SUB:                  dispatch_single_double<ops::Sub>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::MUL:                  dispatch_single_double<ops::Mul>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::DIV:                  dispatch_single_double<ops::Div>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::TRUE_DIV:             dispatch_single_double<ops::TrueDiv>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::FLOOR_DIV:            dispatch_single_double<ops::FloorDiv>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::MOD:                  dispatch_single_double<ops::Mod>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::PYMOD:                dispatch_single_double<ops::PyMod>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::POW:                  dispatch_single_double<ops::Pow>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::EQUAL:                //dispatch_single_double<ops::Equal>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NOT_EQUAL:            //dispatch_single_double<ops::NotEqual>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NULL_EQUALS:          //dispatch_single_double<ops::NullEquals>(*outd, *lhsd, *rhsd, stream); break;
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_equality_op(*outd, *lhsd, *rhsd, op, stream); break;
    case binary_operator::LESS:                 //dispatch_single_double<ops::Less>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::GREATER:              //dispatch_single_double<ops::Greater>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LESS_EQUAL:           //dispatch_single_double<ops::LessEqual>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::GREATER_EQUAL:        //dispatch_single_double<ops::GreaterEqual>(*outd, *lhsd, *rhsd, stream); break;
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_comparison_op(*outd, *lhsd, *rhsd, op, stream); break;
    case binary_operator::BITWISE_AND:          dispatch_single_double<ops::BitwiseAnd>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::BITWISE_OR:           dispatch_single_double<ops::BitwiseOr>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::BITWISE_XOR:          dispatch_single_double<ops::BitwiseXor>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOGICAL_AND:          dispatch_single_double<ops::LogicalAnd>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOGICAL_OR:           dispatch_single_double<ops::LogicalOr>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_LEFT:           dispatch_single_double<ops::ShiftLeft>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_RIGHT:          dispatch_single_double<ops::ShiftRight>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::SHIFT_RIGHT_UNSIGNED: dispatch_single_double<ops::ShiftRightUnsigned>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::LOG_BASE:             dispatch_single_double<ops::LogBase>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::ATAN2:                dispatch_single_double<ops::ATan2>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::PMOD:                 dispatch_single_double<ops::PMod>(*outd, *lhsd, *rhsd, stream); break;
    /*
    case binary_operator::NULL_MAX:             dispatch_single_double<ops::NullMax>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::NULL_MIN:             dispatch_single_double<ops::NullMin>(*outd, *lhsd, *rhsd, stream); break;
    case binary_operator::GENERIC_BINARY:       dispatch_single_double<ops::UserDefinedOp>(*outd, *lhsd, *rhsd, stream); break;
    */
    default:;
  }
  // clang-format on
}

void binary_operation_compiled(mutable_column_view& out,
                               column_view const& lhs,
                               column_view const& rhs,
                               binary_operator op,
                               rmm::cuda_stream_view stream)
{
  // if (is_null_dependent(op)) {
  //  CUDF_FAIL("Unsupported yet");
  // TODO cudf::binops::jit::kernel_v_v_with_validity
  //} else {
  operator_dispatcher(out, lhs, rhs, op, stream);
  //"cudf::binops::jit::kernel_v_v")  //TODO v_s, s_v.
  //}
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

  // if (lhs.type().id() == type_id::STRING and rhs.type().id() == type_id::STRING)
  //  return binops::compiled::binary_operation(lhs, rhs, op, output_type, stream, mr);

  // TODO check if scale conversion required?
  // if (is_fixed_point(lhs.type()) or is_fixed_point(rhs.type()))
  //  CUDF_FAIL("Not yet supported fixed_point");
  // return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  // Check for datatype
  // CUDF_EXPECTS(is_fixed_width(output_type), "Invalid/Unsupported output datatype");
  // CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  // CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

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
