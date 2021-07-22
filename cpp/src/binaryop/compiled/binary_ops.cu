/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace binops {
namespace compiled {

namespace {
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
// specialization for cudf::string_view
template <>
scalar_as_column_device_view::return_type
scalar_as_column_device_view::operator()<cudf::string_view>(scalar const& s,
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

/**
 * @brief Converts scalar to column_device_view with single element.
 *
 * @param scal    scalar to convert
 * @param stream  CUDA stream used for device memory operations and kernel launches.
 * @param mr      Device memory resource used to allocate the returned column's device memory
 * @return        pair with column_device_view and column containing any auxilary data to create
 * column_view from scalar
 */
auto scalar_to_column_device_view(
  scalar const& scal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return type_dispatcher(scal.type(), scalar_as_column_device_view{}, scal, stream, mr);
}

// This functor does the actual comparison between string column value and a scalar string
// or between two string column values using a comparator
template <typename LhsDeviceViewT, typename RhsDeviceViewT, typename OutT, typename CompareFunc>
struct compare_functor {
  LhsDeviceViewT const lhs_dev_view_;  // Scalar or a column device view - lhs
  RhsDeviceViewT const rhs_dev_view_;  // Scalar or a column device view - rhs
  CompareFunc const cfunc_;            // Comparison function

  compare_functor(LhsDeviceViewT const& lhs_dev_view,
                  RhsDeviceViewT const& rhs_dev_view,
                  CompareFunc cf)
    : lhs_dev_view_(lhs_dev_view), rhs_dev_view_(rhs_dev_view), cfunc_(cf)
  {
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_same<LhsViewT, column_device_view>::value &&
                                !std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid() ? rhs_dev_view_.value() : cudf::string_view{});
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<!std::is_same<LhsViewT, column_device_view>::value &&
                                std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid() ? lhs_dev_view_.value() : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }

  // This is used to compare 2 column values
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  CUDA_DEVICE_CALLABLE
    typename std::enable_if_t<std::is_same<LhsViewT, column_device_view>::value &&
                                std::is_same<RhsViewT, column_device_view>::value,
                              OutT>
    operator()(cudf::size_type i) const
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }
};

// This functor performs null aware binop between two columns or a column and a scalar by
// iterating over them on the device
struct null_considering_binop {
  auto get_device_view(cudf::scalar const& scalar_item) const
  {
    return get_scalar_device_view(
      static_cast<cudf::scalar_type_t<cudf::string_view>&>(const_cast<scalar&>(scalar_item)));
  }

  auto get_device_view(column_device_view const& col_item) const { return col_item; }

  template <typename LhsViewT, typename RhsViewT, typename OutT, typename CompareFunc>
  void populate_out_col(LhsViewT const& lhsv,
                        RhsViewT const& rhsv,
                        cudf::size_type col_size,
                        rmm::cuda_stream_view stream,
                        CompareFunc cfunc,
                        OutT* out_col) const
  {
    // Create binop functor instance
    compare_functor<LhsViewT, RhsViewT, OutT, CompareFunc> binop_func{lhsv, rhsv, cfunc};

    // Execute it on every element
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(col_size),
                      out_col,
                      binop_func);
  }

  // This is invoked to perform comparison between cudf string types
  template <typename LhsT, typename RhsT>
  std::unique_ptr<column> operator()(LhsT const& lhs,
                                     RhsT const& rhs,
                                     binary_operator op,
                                     data_type output_type,
                                     cudf::size_type col_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    std::unique_ptr<column> out;
    // Create device views for inputs
    auto const lhs_dev_view = get_device_view(lhs);
    auto const rhs_dev_view = get_device_view(rhs);
    // Validate input
    CUDF_EXPECTS(output_type.id() == lhs.type().id(),
                 "Output column type should match input column type");

    // Shallow copy of the resultant strings
    rmm::device_uvector<cudf::string_view> out_col_strings(col_size, stream);

    // Invalid output column strings - null rows
    cudf::string_view const invalid_str{nullptr, 0};

    // Create a compare function lambda
    auto minmax_func =
      [op, invalid_str] __device__(
        bool lhs_valid, bool rhs_valid, cudf::string_view lhs_value, cudf::string_view rhs_value) {
        if (!lhs_valid && !rhs_valid)
          return invalid_str;
        else if (lhs_valid && rhs_valid) {
          return (op == binary_operator::NULL_MAX)
                   ? thrust::maximum<cudf::string_view>()(lhs_value, rhs_value)
                   : thrust::minimum<cudf::string_view>()(lhs_value, rhs_value);
        } else if (lhs_valid)
          return lhs_value;
        else
          return rhs_value;
      };

    // Populate output column
    populate_out_col(
      lhs_dev_view, rhs_dev_view, col_size, stream, minmax_func, out_col_strings.data());

    // Create an output column with the resultant strings
    return cudf::make_strings_column(out_col_strings, invalid_str, stream, mr);
  }
};

}  // namespace

std::unique_ptr<column> string_null_min_max(scalar const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (rhs.is_empty()) return cudf::make_empty_column(output_type);
  auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
  return null_considering_binop{}(lhs, *rhs_device_view, op, output_type, rhs.size(), stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            scalar const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (lhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
  return null_considering_binop{}(*lhs_device_view, rhs, op, output_type, lhs.size(), stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes do not match");
  if (lhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
  auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
  return null_considering_binop{}(
    *lhs_device_view, *rhs_device_view, op, output_type, lhs.size(), stream, mr);
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
case binary_operator::LESS:                 apply_binary_op<ops::Less>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::GREATER:              apply_binary_op<ops::Greater>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::LESS_EQUAL:           apply_binary_op<ops::LessEqual>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::GREATER_EQUAL:        apply_binary_op<ops::GreaterEqual>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
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
  auto [lhsd, aux] = scalar_to_column_device_view(lhs, stream);
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
  auto [rhsd, aux] = scalar_to_column_device_view(rhs, stream);
  auto outd        = mutable_column_device_view::create(out, stream);
  operator_dispatcher(*outd, *lhsd, *rhsd, false, true, op, stream);
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
