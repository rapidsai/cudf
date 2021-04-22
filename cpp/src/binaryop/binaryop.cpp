/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include "compiled/binary_ops.hpp"
#include "jit/util.hpp"

#include <jit_preprocessed_files/binaryop/jit/kernel.cu.jit.hpp>

#include <jit/cache.hpp>
#include <jit/parser.hpp>
#include <jit/type.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string>

#include <thrust/optional.h>

namespace cudf {

namespace binops {
namespace detail {

/**
 * @brief Computes output valid mask for op between a column and a scalar
 */
rmm::device_buffer scalar_col_valid_mask_and(column_view const& col,
                                             scalar const& s,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  if (col.is_empty()) return rmm::device_buffer{0, stream, mr};

  if (not s.is_valid()) {
    return cudf::detail::create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr);
  } else if (s.is_valid() and col.nullable()) {
    return cudf::detail::copy_bitmask(col, stream, mr);
  } else {
    return rmm::device_buffer{0, stream, mr};
  }
}
}  // namespace detail

namespace jit {

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      OperatorType op_type,
                      rmm::cuda_stream_view stream)
{
  if (is_null_dependent(op)) {
    std::string kernel_name =
      jitify2::reflection::Template("cudf::binops::jit::kernel_v_s_with_validity")  //
        .instantiate(cudf::jit::get_type_name(out.type()),  // list of template arguments
                     cudf::jit::get_type_name(lhs.type()),
                     cudf::jit::get_type_name(rhs.type()),
                     get_operator_name(op, op_type));

    cudf::jit::get_program_cache(*binaryop_jit_kernel_cu_jit)
      .get_kernel(kernel_name, {}, {}, {"-arch=sm_."})       //
      ->configure_1d_max_occupancy(0, 0, 0, stream.value())  //
      ->launch(out.size(),
               cudf::jit::get_data_ptr(out),
               cudf::jit::get_data_ptr(lhs),
               cudf::jit::get_data_ptr(rhs),
               out.null_mask(),
               lhs.null_mask(),
               lhs.offset(),
               rhs.is_valid());
  } else {
    std::string kernel_name =
      jitify2::reflection::Template("cudf::binops::jit::kernel_v_s")  //
        .instantiate(cudf::jit::get_type_name(out.type()),            // list of template arguments
                     cudf::jit::get_type_name(lhs.type()),
                     cudf::jit::get_type_name(rhs.type()),
                     get_operator_name(op, op_type));

    cudf::jit::get_program_cache(*binaryop_jit_kernel_cu_jit)
      .get_kernel(kernel_name, {}, {}, {"-arch=sm_."})       //
      ->configure_1d_max_occupancy(0, 0, 0, stream.value())  //
      ->launch(out.size(),
               cudf::jit::get_data_ptr(out),
               cudf::jit::get_data_ptr(lhs),
               cudf::jit::get_data_ptr(rhs));
  }
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  return binary_operation(out, lhs, rhs, op, OperatorType::Direct, stream);
}

void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  return binary_operation(out, rhs, lhs, op, OperatorType::Reverse, stream);
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  if (is_null_dependent(op)) {
    std::string kernel_name =
      jitify2::reflection::Template("cudf::binops::jit::kernel_v_v_with_validity")  //
        .instantiate(cudf::jit::get_type_name(out.type()),  // list of template arguments
                     cudf::jit::get_type_name(lhs.type()),
                     cudf::jit::get_type_name(rhs.type()),
                     get_operator_name(op, OperatorType::Direct));

    cudf::jit::get_program_cache(*binaryop_jit_kernel_cu_jit)
      .get_kernel(kernel_name, {}, {}, {"-arch=sm_."})       //
      ->configure_1d_max_occupancy(0, 0, 0, stream.value())  //
      ->launch(out.size(),
               cudf::jit::get_data_ptr(out),
               cudf::jit::get_data_ptr(lhs),
               cudf::jit::get_data_ptr(rhs),
               out.null_mask(),
               lhs.null_mask(),
               rhs.offset(),
               rhs.null_mask(),
               rhs.offset());
  } else {
    std::string kernel_name =
      jitify2::reflection::Template("cudf::binops::jit::kernel_v_v")  //
        .instantiate(cudf::jit::get_type_name(out.type()),            // list of template arguments
                     cudf::jit::get_type_name(lhs.type()),
                     cudf::jit::get_type_name(rhs.type()),
                     get_operator_name(op, OperatorType::Direct));

    cudf::jit::get_program_cache(*binaryop_jit_kernel_cu_jit)
      .get_kernel(kernel_name, {}, {}, {"-arch=sm_."})       //
      ->configure_1d_max_occupancy(0, 0, 0, stream.value())  //
      ->launch(out.size(),
               cudf::jit::get_data_ptr(out),
               cudf::jit::get_data_ptr(lhs),
               cudf::jit::get_data_ptr(rhs));
  }
}

void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      const std::string& ptx,
                      rmm::cuda_stream_view stream)
{
  std::string const output_type_name = cudf::jit::get_type_name(out.type());

  std::string ptx_hash =
    "prog_binop." + std::to_string(std::hash<std::string>{}(ptx + output_type_name));
  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(ptx, "GENERIC_BINARY_OP", output_type_name);

  std::string kernel_name =
    jitify2::reflection::Template("cudf::binops::jit::kernel_v_v")  //
      .instantiate(output_type_name,                                // list of template arguments
                   cudf::jit::get_type_name(lhs.type()),
                   cudf::jit::get_type_name(rhs.type()),
                   get_operator_name(binary_operator::GENERIC_BINARY, OperatorType::Direct));

  cudf::jit::get_program_cache(*binaryop_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"binaryop/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, 0, stream.value())                                  //
    ->launch(out.size(),
             cudf::jit::get_data_ptr(out),
             cudf::jit::get_data_ptr(lhs),
             cudf::jit::get_data_ptr(rhs));
}

}  // namespace jit
}  // namespace binops

namespace detail {

// There are 3 overloads of each of the following functions:
// - `make_fixed_width_column_for_output`
// - `fixed_point_binary_operation`
// - `binary_operation`

// The overloads are overloaded on the first two parameters of each function:
// - scalar      const& lhs, column_view const& rhs,
// - column_view const& lhs, scalar      const& rhs
// - column_view const& lhs, column_view const& rhs,

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `scalar` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(scalar const& lhs,
                                                           column_view const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, rhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto new_mask = binops::detail::scalar_col_valid_mask_and(rhs, lhs, stream, mr);
    return make_fixed_width_column(
      output_type, rhs.size(), std::move(new_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  }
};

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `scalar` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(column_view const& lhs,
                                                           scalar const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, lhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto new_mask = binops::detail::scalar_col_valid_mask_and(lhs, rhs, stream, mr);
    return make_fixed_width_column(
      output_type, lhs.size(), std::move(new_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  }
};

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(column_view const& lhs,
                                                           column_view const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, rhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto new_mask = cudf::detail::bitmask_and(table_view({lhs, rhs}), stream, mr);
    return make_fixed_width_column(
      output_type, lhs.size(), std::move(new_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);
  }
};

/**
 * @brief Returns `true` if `binary_operator` `op` is a basic arithmetic binary operation
 */
bool is_basic_arithmetic_binop(binary_operator op)
{
  return op == binary_operator::ADD or       // operator +
         op == binary_operator::SUB or       // operator -
         op == binary_operator::MUL or       // operator *
         op == binary_operator::DIV or       // operator / using common type of lhs and rhs
         op == binary_operator::NULL_MIN or  // 2 null = null, 1 null = value, else min
         op == binary_operator::NULL_MAX;    // 2 null = null, 1 null = value, else max
}

/**
 * @brief Returns `true` if `binary_operator` `op` is a comparison binary operation
 */
bool is_comparison_binop(binary_operator op)
{
  return op == binary_operator::EQUAL or          // operator ==
         op == binary_operator::NOT_EQUAL or      // operator !=
         op == binary_operator::LESS or           // operator <
         op == binary_operator::GREATER or        // operator >
         op == binary_operator::LESS_EQUAL or     // operator <=
         op == binary_operator::GREATER_EQUAL or  // operator >=
         op == binary_operator::NULL_EQUALS;      // 2 null = true; 1 null = false; else ==
}

/**
 * @brief Returns `true` if `binary_operator` `op` is supported by `fixed_point`
 */
bool is_supported_fixed_point_binop(binary_operator op)
{
  return is_basic_arithmetic_binop(op) or is_comparison_binop(op);
}

/**
 * @brief Helper predicate function that identifies if `op` requires scales to be the same
 *
 * @param op `binary_operator`
 * @return true `op` requires scales of lhs and rhs to be the same
 * @return false `op` does not require scales of lhs and rhs to be the same
 */
bool is_same_scale_necessary(binary_operator op)
{
  return op != binary_operator::MUL && op != binary_operator::DIV;
}

template <typename Lhs, typename Rhs>
void fixed_point_binary_operation_validation(binary_operator op,
                                             Lhs lhs,
                                             Rhs rhs,
                                             thrust::optional<cudf::data_type> output_type = {})
{
  CUDF_EXPECTS(is_fixed_point(lhs), "Input must have fixed_point data_type.");
  CUDF_EXPECTS(is_fixed_point(rhs), "Input must have fixed_point data_type.");
  CUDF_EXPECTS(is_supported_fixed_point_binop(op), "Unsupported fixed_point binary operation");
  CUDF_EXPECTS(lhs.id() == rhs.id(), "Data type mismatch");
  if (output_type.has_value()) {
    if (is_comparison_binop(op))
      CUDF_EXPECTS(output_type == cudf::data_type{type_id::BOOL8},
                   "Comparison operations require boolean output type.");
    else
      CUDF_EXPECTS(is_fixed_point(output_type.value()),
                   "fixed_point binary operations require fixed_point output type.");
  }
}

/**
 * @brief Function to compute binary operation of one `column_view` and one `scalar`
 *
 * @param lhs Left-hand side `scalar` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Resulting output column from the binary operation
 */
std::unique_ptr<column> fixed_point_binary_operation(scalar const& lhs,
                                                     column_view const& rhs,
                                                     binary_operator op,
                                                     cudf::data_type output_type,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  using namespace numeric;

  fixed_point_binary_operation_validation(op, lhs.type(), rhs.type(), output_type);

  if (rhs.is_empty())
    return make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  auto const scale = binary_operation_fixed_point_scale(op, lhs.type().scale(), rhs.type().scale());
  auto const type =
    is_comparison_binop(op) ? data_type{type_id::BOOL8} : cudf::data_type{rhs.type().id(), scale};
  auto out      = make_fixed_width_column_for_output(lhs, rhs, op, type, stream, mr);
  auto out_view = out->mutable_view();

  if (lhs.type().scale() != rhs.type().scale() && is_same_scale_necessary(op)) {
    // Adjust scalar/column so they have they same scale
    if (rhs.type().scale() < lhs.type().scale()) {
      auto const diff = lhs.type().scale() - rhs.type().scale();
      if (lhs.type().id() == type_id::DECIMAL32) {
        auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
        auto const val    = static_cast<fixed_point_scalar<decimal32> const&>(lhs).value();
        auto const scale  = scale_type{rhs.type().scale()};
        auto const scalar = make_fixed_point_scalar<decimal32>(val * factor, scale);
        binops::jit::binary_operation(out_view, *scalar, rhs, op, stream);
      } else {
        CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
        auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
        auto const val    = static_cast<fixed_point_scalar<decimal64> const&>(lhs).value();
        auto const scale  = scale_type{rhs.type().scale()};
        auto const scalar = make_fixed_point_scalar<decimal64>(val * factor, scale);
        binops::jit::binary_operation(out_view, *scalar, rhs, op, stream);
      }
    } else {
      auto const diff   = rhs.type().scale() - lhs.type().scale();
      auto const result = [&] {
        if (lhs.type().id() == type_id::DECIMAL32) {
          auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal32>(factor, scale_type{-diff});
          return binary_operation(*scalar, rhs, binary_operator::MUL, lhs.type(), stream, mr);
        } else {
          CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
          auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal64>(factor, scale_type{-diff});
          return binary_operation(*scalar, rhs, binary_operator::MUL, lhs.type(), stream, mr);
        }
      }();
      binops::jit::binary_operation(out_view, lhs, result->view(), op, stream);
    }
  } else {
    binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  }
  return output_type.scale() != scale ? cudf::cast(out_view, output_type) : std::move(out);
}

/**
 * @brief Function to compute binary operation of one `column_view` and one `scalar`
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `scalar` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Resulting output column from the binary operation
 */
std::unique_ptr<column> fixed_point_binary_operation(column_view const& lhs,
                                                     scalar const& rhs,
                                                     binary_operator op,
                                                     cudf::data_type output_type,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  using namespace numeric;

  fixed_point_binary_operation_validation(op, lhs.type(), rhs.type(), output_type);

  if (lhs.is_empty())
    return make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  auto const scale = binary_operation_fixed_point_scale(op, lhs.type().scale(), rhs.type().scale());
  auto const type =
    is_comparison_binop(op) ? data_type{type_id::BOOL8} : cudf::data_type{lhs.type().id(), scale};
  auto out      = make_fixed_width_column_for_output(lhs, rhs, op, type, stream, mr);
  auto out_view = out->mutable_view();

  if (lhs.type().scale() != rhs.type().scale() && is_same_scale_necessary(op)) {
    // Adjust scalar/column so they have they same scale
    if (rhs.type().scale() > lhs.type().scale()) {
      auto const diff = rhs.type().scale() - lhs.type().scale();
      if (rhs.type().id() == type_id::DECIMAL32) {
        auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
        auto const val    = static_cast<fixed_point_scalar<decimal32> const&>(rhs).value();
        auto const scale  = scale_type{lhs.type().scale()};
        auto const scalar = make_fixed_point_scalar<decimal32>(val * factor, scale);
        binops::jit::binary_operation(out_view, lhs, *scalar, op, stream);
      } else {
        CUDF_EXPECTS(rhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
        auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
        auto const val    = static_cast<fixed_point_scalar<decimal64> const&>(rhs).value();
        auto const scale  = scale_type{rhs.type().scale()};
        auto const scalar = make_fixed_point_scalar<decimal64>(val * factor, scale);
        binops::jit::binary_operation(out_view, lhs, *scalar, op, stream);
      }
    } else {
      auto const diff   = lhs.type().scale() - rhs.type().scale();
      auto const result = [&] {
        if (rhs.type().id() == type_id::DECIMAL32) {
          auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal32>(factor, scale_type{-diff});
          return binary_operation(*scalar, lhs, binary_operator::MUL, rhs.type(), stream, mr);
        } else {
          CUDF_EXPECTS(rhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
          auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal64>(factor, scale_type{-diff});
          return binary_operation(*scalar, lhs, binary_operator::MUL, rhs.type(), stream, mr);
        }
      }();
      binops::jit::binary_operation(out_view, result->view(), rhs, op, stream);
    }
  } else {
    binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  }
  return output_type.scale() != scale ? cudf::cast(out_view, output_type) : std::move(out);
}

/**
 * @brief Function to compute binary operation of two `column_view`s
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param mr Device memory resource to use for device memory allocation
 * @param stream CUDA stream used for device memory operations
 * @return std::unique_ptr<column> Resulting output column from the binary operation
 */
std::unique_ptr<column> fixed_point_binary_operation(column_view const& lhs,
                                                     column_view const& rhs,
                                                     binary_operator op,
                                                     cudf::data_type output_type,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  using namespace numeric;

  fixed_point_binary_operation_validation(op, lhs.type(), rhs.type(), output_type);

  if (lhs.is_empty() or rhs.is_empty())
    return make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  auto const scale = binary_operation_fixed_point_scale(op, lhs.type().scale(), rhs.type().scale());
  auto const type =
    is_comparison_binop(op) ? data_type{type_id::BOOL8} : cudf::data_type{lhs.type().id(), scale};
  auto out      = make_fixed_width_column_for_output(lhs, rhs, op, type, stream, mr);
  auto out_view = out->mutable_view();

  if (lhs.type().scale() != rhs.type().scale() && is_same_scale_necessary(op)) {
    if (rhs.type().scale() < lhs.type().scale()) {
      auto const diff   = lhs.type().scale() - rhs.type().scale();
      auto const result = [&] {
        if (lhs.type().id() == type_id::DECIMAL32) {
          auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal32>(factor, scale_type{-diff});
          return binary_operation(*scalar, lhs, binary_operator::MUL, rhs.type(), stream, mr);
        } else {
          CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
          auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal64>(factor, scale_type{-diff});
          return binary_operation(*scalar, lhs, binary_operator::MUL, rhs.type(), stream, mr);
        }
      }();
      binops::jit::binary_operation(out_view, result->view(), rhs, op, stream);
    } else {
      auto const diff   = rhs.type().scale() - lhs.type().scale();
      auto const result = [&] {
        if (lhs.type().id() == type_id::DECIMAL32) {
          auto const factor = numeric::detail::ipow<int32_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal32>(factor, scale_type{-diff});
          return binary_operation(*scalar, rhs, binary_operator::MUL, lhs.type(), stream, mr);
        } else {
          CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL64, "Unexpected DTYPE");
          auto const factor = numeric::detail::ipow<int64_t, Radix::BASE_10>(diff);
          auto const scalar = make_fixed_point_scalar<decimal64>(factor, scale_type{-diff});
          return binary_operation(*scalar, rhs, binary_operator::MUL, lhs.type(), stream, mr);
        }
      }();
      binops::jit::binary_operation(out_view, lhs, result->view(), op, stream);
    }
  } else {
    binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  }
  return output_type.scale() != scale ? cudf::cast(out_view, output_type) : std::move(out);
}

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  if (lhs.type().id() == type_id::STRING and rhs.type().id() == type_id::STRING)
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, stream, mr);

  if (is_fixed_point(lhs.type()) or is_fixed_point(rhs.type()))
    return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type), "Invalid/Unsupported output datatype");
  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if (rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  if (lhs.type().id() == type_id::STRING and rhs.type().id() == type_id::STRING)
    return binops::compiled::binary_operation(lhs, rhs, op, output_type, stream, mr);

  if (is_fixed_point(lhs.type()) or is_fixed_point(rhs.type()))
    return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type), "Invalid/Unsupported output datatype");
  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if (lhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
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
    return fixed_point_binary_operation(lhs, rhs, op, output_type, stream, mr);

  // Check for datatype
  CUDF_EXPECTS(is_fixed_width(output_type), "Invalid/Unsupported output datatype");
  CUDF_EXPECTS(is_fixed_width(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_fixed_width(rhs.type()), "Invalid/Unsupported rhs datatype");

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if (lhs.is_empty() or rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, op, stream);
  return out;
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  // Check for datatype
  auto is_type_supported_ptx = [](data_type type) -> bool {
    return is_fixed_width(type) and not is_fixed_point(type) and
           type.id() != type_id::INT8;  // Numba PTX doesn't support int8
  };

  CUDF_EXPECTS(is_type_supported_ptx(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(rhs.type()), "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(output_type), "Invalid/Unsupported output datatype");

  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  auto new_mask = bitmask_and(table_view({lhs, rhs}), stream, mr);
  auto out      = make_fixed_width_column(
    output_type, lhs.size(), std::move(new_mask), cudf::UNKNOWN_NULL_COUNT, stream, mr);

  // Check for 0 sized data
  if (lhs.is_empty() or rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ptx, stream);
  return out;
}

}  // namespace detail

int32_t binary_operation_fixed_point_scale(binary_operator op,
                                           int32_t left_scale,
                                           int32_t right_scale)
{
  CUDF_EXPECTS(cudf::detail::is_supported_fixed_point_binop(op),
               "Unsupported fixed_point binary operation.");
  if (op == binary_operator::MUL) return left_scale + right_scale;
  if (op == binary_operator::DIV) return left_scale - right_scale;
  return std::min(left_scale, right_scale);
}

cudf::data_type binary_operation_fixed_point_output_type(binary_operator op,
                                                         cudf::data_type const& lhs,
                                                         cudf::data_type const& rhs)
{
  cudf::detail::fixed_point_binary_operation_validation(op, lhs, rhs);

  auto const scale = binary_operation_fixed_point_scale(op, lhs.scale(), rhs.scale());
  return cudf::data_type{lhs.id(), scale};
}

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, ptx, output_type, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
