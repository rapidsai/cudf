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
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace binops {
namespace compiled {
/**
 * @brief Converts scalar to column_device_view with single element.
 *
 * @return pair with column_device_view and column containing any auxilary data to create
 * column_view from scalar
 */
struct scalar_as_column_device_view {
  using return_type = typename std::pair<decltype(column_device_view::create(column_view{})),
                                         std::unique_ptr<column>>;
  template <typename T, std::enable_if_t<(is_fixed_width<T>())>* = nullptr>
  return_type operator()(scalar const& s,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  {
    auto h_scalar_type_view = static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(s));
    auto col_v =
      column_view(s.type(), 1, h_scalar_type_view.data(), (bitmask_type const*)s.validity_data());
    return std::pair{column_device_view::create(col_v, stream), std::unique_ptr<column>(nullptr)};
  }
  template <typename T, std::enable_if_t<(!is_fixed_width<T>())>* = nullptr>
  return_type operator()(scalar const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Unsupported type");
  }
};
// specialization for string_view
template <>
scalar_as_column_device_view::return_type scalar_as_column_device_view::
operator()<cudf::string_view>(scalar const& s,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  using T                 = cudf::string_view;
  auto h_scalar_type_view = static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(s));

  // build offsets column from the string size
  auto offsets_transformer_itr =
    thrust::make_constant_iterator<size_type>(h_scalar_type_view.size());
  auto offsets_column = strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + 1, stream, mr);

  auto chars_column_v =
    column_view(data_type{type_id::INT8}, h_scalar_type_view.size(), h_scalar_type_view.data());
  // Construct string column_view
  auto col_v = column_view(s.type(),
                           1,
                           nullptr,
                           (bitmask_type const*)s.validity_data(),
                           cudf::UNKNOWN_NULL_COUNT,
                           0,
                           {offsets_column->view(), chars_column_v});
  return std::pair{column_device_view::create(col_v, stream), std::move(offsets_column)};
}

void operator_dispatcher(mutable_column_device_view& out,
                         column_device_view const& lhs,
                         column_device_view const& rhs,
                         bool is_lhs_scalar,
                         bool is_rhs_scalar,
                         binary_operator op,
                         rmm::cuda_stream_view stream)
{
  // clang-format off
  switch (op) {
    case binary_operator::ADD:                  apply_binary_op<ops::Add>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::SUB:                  apply_binary_op<ops::Sub>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::MUL:                  apply_binary_op<ops::Mul>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::DIV:                  apply_binary_op<ops::Div>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::TRUE_DIV:             apply_binary_op<ops::TrueDiv>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::FLOOR_DIV:            apply_binary_op<ops::FloorDiv>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::MOD:                  apply_binary_op<ops::Mod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::PYMOD:                apply_binary_op<ops::PyMod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::POW:                  apply_binary_op<ops::Pow>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::EQUAL:
    case binary_operator::NOT_EQUAL:
    case binary_operator::NULL_EQUALS:
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_equality_op(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, op, stream); break;
    case binary_operator::LESS:
    case binary_operator::GREATER:
    case binary_operator::LESS_EQUAL:
    case binary_operator::GREATER_EQUAL:
      if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
        dispatch_comparison_op(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, op, stream); break;
    case binary_operator::BITWISE_AND:          apply_binary_op<ops::BitwiseAnd>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::BITWISE_OR:           apply_binary_op<ops::BitwiseOr>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::BITWISE_XOR:          apply_binary_op<ops::BitwiseXor>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::LOGICAL_AND:          apply_binary_op<ops::LogicalAnd>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::LOGICAL_OR:           apply_binary_op<ops::LogicalOr>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    /*
    case binary_operator::GENERIC_BINARY:      // Cannot be compiled, should be called by jit::binary_operation
    */
    case binary_operator::SHIFT_LEFT:           apply_binary_op<ops::ShiftLeft>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::SHIFT_RIGHT:          apply_binary_op<ops::ShiftRight>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::SHIFT_RIGHT_UNSIGNED: apply_binary_op<ops::ShiftRightUnsigned>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::LOG_BASE:             apply_binary_op<ops::LogBase>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::ATAN2:                apply_binary_op<ops::ATan2>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::PMOD:                 apply_binary_op<ops::PMod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::NULL_MAX:             apply_binary_op<ops::NullMax>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    case binary_operator::NULL_MIN:             apply_binary_op<ops::NullMin>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
    default:;
  }
  // clang-format on
}

// vector_vector
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);
  operator_dispatcher(*outd, *lhsd, *rhsd, false, false, op, stream);
}
// scalar_vector
void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  auto [lhsd, aux] = type_dispatcher(lhs.type(),
                                     scalar_as_column_device_view{},
                                     lhs,
                                     stream,
                                     rmm::mr::get_current_device_resource());
  auto rhsd        = column_device_view::create(rhs, stream);
  auto outd        = mutable_column_device_view::create(out, stream);
  operator_dispatcher(*outd, *lhsd, *rhsd, true, false, op, stream);
}
// vector_scalar
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  auto lhsd        = column_device_view::create(lhs, stream);
  auto [rhsd, aux] = type_dispatcher(rhs.type(),
                                     scalar_as_column_device_view{},
                                     rhs,
                                     stream,
                                     rmm::mr::get_current_device_resource());
  auto outd        = mutable_column_device_view::create(out, stream);
  operator_dispatcher(*outd, *lhsd, *rhsd, false, true, op, stream);
}
}  // namespace compiled
}  // namespace binops
}  // namespace cudf
