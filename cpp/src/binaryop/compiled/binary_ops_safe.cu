/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

#include <cstdint>

namespace cudf {
namespace binops {
namespace compiled {
namespace {

// Functor that performs the decimal-safe binary op for one row.
//
// Both operands and the output are assumed to use the same base-10 decimal storage type
// (caller-validated). We load the raw integer rep, wrap it into a `Track::on` value type so
// arithmetic propagates a sticky overflow bit, run the op, rescale to the output scale, and
// atomicMax the global flag if any active row overflows.
template <typename SafeDecimal, typename Op>
struct decimal_safe_op_kernel {
  using Rep = typename SafeDecimal::rep;

  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;
  numeric::scale_type out_scale;
  unsigned int* d_overflow;

  __device__ __forceinline__ void operator()(size_type i) const
  {
    auto const li         = is_lhs_scalar ? 0 : i;
    auto const ri         = is_rhs_scalar ? 0 : i;
    bool const row_active = lhs.is_valid(li) && rhs.is_valid(ri);

    auto const lscale = numeric::scale_type{lhs.type().scale()};
    auto const rscale = numeric::scale_type{rhs.type().scale()};

    SafeDecimal const x{numeric::scaled_integer<Rep>{lhs.element<Rep>(li), lscale}};
    SafeDecimal const y{numeric::scaled_integer<Rep>{rhs.element<Rep>(ri), rscale}};

    SafeDecimal r = Op{}(x, y);
    bool bad      = r.overflow_occurred();

    r   = r.rescaled(out_scale);
    bad = bad || r.overflow_occurred();

    if (row_active && bad) { atomicMax(d_overflow, 1u); }
    out.data<Rep>()[i] = r.value();
  }
};

template <typename SafeDecimal, typename Op>
void launch_decimal_safe_kernel(mutable_column_device_view& outd,
                                column_device_view const& lhsd,
                                column_device_view const& rhsd,
                                bool is_lhs_scalar,
                                bool is_rhs_scalar,
                                size_type n,
                                unsigned int* d_overflow,
                                rmm::cuda_stream_view stream)
{
  auto const out_scale = numeric::scale_type{outd.type().scale()};
  decimal_safe_op_kernel<SafeDecimal, Op> kern{
    outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, out_scale, d_overflow};
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream), cuda::counting_iterator<size_type>{0}, n, kern);
}

template <typename SafeDecimal>
void dispatch_op_and_run(mutable_column_device_view& outd,
                         column_device_view const& lhsd,
                         column_device_view const& rhsd,
                         bool is_lhs_scalar,
                         bool is_rhs_scalar,
                         size_type n,
                         unsigned int* d_overflow,
                         binary_operator op,
                         rmm::cuda_stream_view stream)
{
  switch (op) {
    case binary_operator::ADD:
      launch_decimal_safe_kernel<SafeDecimal, ops::Add>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::SUB:
      launch_decimal_safe_kernel<SafeDecimal, ops::Sub>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::MUL:
      launch_decimal_safe_kernel<SafeDecimal, ops::Mul>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::DIV:
      launch_decimal_safe_kernel<SafeDecimal, ops::Div>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::MOD:
      launch_decimal_safe_kernel<SafeDecimal, ops::Mod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::PMOD:
      launch_decimal_safe_kernel<SafeDecimal, ops::PMod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    case binary_operator::PYMOD:
      launch_decimal_safe_kernel<SafeDecimal, ops::PyMod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow, stream);
      break;
    default:
      CUDF_FAIL("binary_operation_safe only supports ADD, SUB, MUL, DIV, MOD, PMOD, and PYMOD.");
  }
}

}  // namespace

void apply_binary_op_safe(mutable_column_view& out,
                          column_view const& lhs,
                          column_view const& rhs,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          binary_operator op,
                          unsigned int* d_overflow_flag,
                          rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(d_overflow_flag != nullptr,
               "binary_operation_safe requires a non-null device overflow flag.");
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id() && lhs.type().id() == out.type().id(),
               "binary_operation_safe requires lhs/rhs/out to share the same decimal storage type.");

  if (out.size() == 0) { return; }

  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);

  switch (out.type().id()) {
    case type_id::DECIMAL32:
    case type_id::DECIMAL32_SAFE:
      dispatch_op_and_run<numeric::decimal32_safe>(*outd,
                                                   *lhsd,
                                                   *rhsd,
                                                   is_lhs_scalar,
                                                   is_rhs_scalar,
                                                   out.size(),
                                                   d_overflow_flag,
                                                   op,
                                                   stream);
      break;
    case type_id::DECIMAL64:
    case type_id::DECIMAL64_SAFE:
      dispatch_op_and_run<numeric::decimal64_safe>(*outd,
                                                   *lhsd,
                                                   *rhsd,
                                                   is_lhs_scalar,
                                                   is_rhs_scalar,
                                                   out.size(),
                                                   d_overflow_flag,
                                                   op,
                                                   stream);
      break;
    case type_id::DECIMAL128:
    case type_id::DECIMAL128_SAFE:
      dispatch_op_and_run<numeric::decimal128_safe>(*outd,
                                                    *lhsd,
                                                    *rhsd,
                                                    is_lhs_scalar,
                                                    is_rhs_scalar,
                                                    out.size(),
                                                    d_overflow_flag,
                                                    op,
                                                    stream);
      break;
    default: CUDF_FAIL("binary_operation_safe requires a base-10 decimal output type.");
  }
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
