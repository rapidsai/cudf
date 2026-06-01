/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/fixed_point/detail/safe_arithmetic.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

namespace cudf {
namespace binops {
namespace compiled {
namespace {

// One thin overflow-aware functor per supported binary operator. Each functor
// wraps the matching free function in `cudf/fixed_point/detail/safe_arithmetic.hpp`
// and returns the `safe_result` directly, so the kernel only needs to OR the
// op-level overflow with the rescale-to-output overflow before recording the
// global flag.
struct SafeAdd {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_add(lhs, rhs);
  }
};
struct SafeSub {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_sub(lhs, rhs);
  }
};
struct SafeMul {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_mul(lhs, rhs);
  }
};
struct SafeDiv {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_div(lhs, rhs);
  }
};
struct SafeMod {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_mod(lhs, rhs);
  }
};
struct SafePMod {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_pmod(lhs, rhs);
  }
};
struct SafePyMod {
  template <typename Decimal>
  __device__ __forceinline__ auto operator()(Decimal lhs, Decimal rhs) const
  {
    return numeric::detail::safe_pymod(lhs, rhs);
  }
};

// Per-row functor that performs an overflow-checked decimal binary op.
//
// Both operands and the output share the same base-10 decimal storage type
// (caller-validated). Each thread loads the raw integer rep, wraps it in a
// regular `fixed_point` value (no sticky-flag layer), calls the matching
// `safe_*` free function, rescales to the output scale via `safe_rescaled`,
// and writes a per-row overflow bit (`true` iff active row && (op or rescale)
// overflowed) to the `d_overflow_per_row` buffer.
template <typename Decimal, typename SafeOp>
struct decimal_safe_op_kernel {
  using Rep = typename Decimal::rep;

  mutable_column_device_view out;
  column_device_view lhs;
  column_device_view rhs;
  bool is_lhs_scalar;
  bool is_rhs_scalar;
  numeric::scale_type out_scale;
  bool* d_overflow_per_row;

  __device__ __forceinline__ void operator()(size_type i) const
  {
    auto const li = is_lhs_scalar ? 0 : i;
    auto const ri = is_rhs_scalar ? 0 : i;
    // Per cuDF convention, null payloads of fixed-width columns are unspecified
    // and must not be read. Short-circuit so we never call `element<Rep>` on a
    // null row, and write defined neutral values into the output's null slots
    // (the result column's null mask, set by the caller, is what makes them null).
    if (!(lhs.is_valid(li) && rhs.is_valid(ri))) {
      d_overflow_per_row[i] = false;
      out.data<Rep>()[i]    = Rep{};
      return;
    }

    auto const lscale = numeric::scale_type{lhs.type().scale()};
    auto const rscale = numeric::scale_type{rhs.type().scale()};

    Decimal const x{numeric::scaled_integer<Rep>{lhs.element<Rep>(li), lscale}};
    Decimal const y{numeric::scaled_integer<Rep>{rhs.element<Rep>(ri), rscale}};

    auto const op_res       = SafeOp{}(x, y);
    auto const rescaled_res = numeric::detail::safe_rescaled(op_res.value, out_scale);

    d_overflow_per_row[i] = op_res.overflow || rescaled_res.overflow;
    out.data<Rep>()[i]    = rescaled_res.value.value();
  }
};

template <typename Decimal, typename SafeOp>
void launch_decimal_safe_kernel(mutable_column_device_view& outd,
                                column_device_view const& lhsd,
                                column_device_view const& rhsd,
                                bool is_lhs_scalar,
                                bool is_rhs_scalar,
                                size_type n,
                                bool* d_overflow_per_row,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto const out_scale = numeric::scale_type{outd.type().scale()};
  decimal_safe_op_kernel<Decimal, SafeOp> kern{
    outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, out_scale, d_overflow_per_row};
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, mr), cuda::counting_iterator<size_type>{0}, n, kern);
}

template <typename Decimal>
void dispatch_op_and_run(mutable_column_device_view& outd,
                         column_device_view const& lhsd,
                         column_device_view const& rhsd,
                         bool is_lhs_scalar,
                         bool is_rhs_scalar,
                         size_type n,
                         bool* d_overflow_per_row,
                         binary_operator op,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  switch (op) {
    case binary_operator::ADD:
      launch_decimal_safe_kernel<Decimal, SafeAdd>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::SUB:
      launch_decimal_safe_kernel<Decimal, SafeSub>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::MUL:
      launch_decimal_safe_kernel<Decimal, SafeMul>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::DIV:
      launch_decimal_safe_kernel<Decimal, SafeDiv>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::MOD:
      launch_decimal_safe_kernel<Decimal, SafeMod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::PMOD:
      launch_decimal_safe_kernel<Decimal, SafePMod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
      break;
    case binary_operator::PYMOD:
      launch_decimal_safe_kernel<Decimal, SafePyMod>(
        outd, lhsd, rhsd, is_lhs_scalar, is_rhs_scalar, n, d_overflow_per_row, stream, mr);
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
                          bool* d_overflow_per_row,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(d_overflow_per_row != nullptr,
               "binary_operation_safe requires a non-null device per-row overflow buffer.");
  CUDF_EXPECTS(
    lhs.type().id() == rhs.type().id() && lhs.type().id() == out.type().id(),
    "binary_operation_safe requires lhs/rhs/out to share the same decimal storage type.");

  if (out.size() == 0) { return; }

  auto lhsd = column_device_view::create(lhs, stream);
  auto rhsd = column_device_view::create(rhs, stream);
  auto outd = mutable_column_device_view::create(out, stream);

  switch (out.type().id()) {
    case type_id::DECIMAL32:
      dispatch_op_and_run<numeric::decimal32>(*outd,
                                              *lhsd,
                                              *rhsd,
                                              is_lhs_scalar,
                                              is_rhs_scalar,
                                              out.size(),
                                              d_overflow_per_row,
                                              op,
                                              stream,
                                              mr);
      break;
    case type_id::DECIMAL64:
      dispatch_op_and_run<numeric::decimal64>(*outd,
                                              *lhsd,
                                              *rhsd,
                                              is_lhs_scalar,
                                              is_rhs_scalar,
                                              out.size(),
                                              d_overflow_per_row,
                                              op,
                                              stream,
                                              mr);
      break;
    case type_id::DECIMAL128:
      dispatch_op_and_run<numeric::decimal128>(*outd,
                                               *lhsd,
                                               *rhsd,
                                               is_lhs_scalar,
                                               is_rhs_scalar,
                                               out.size(),
                                               d_overflow_per_row,
                                               op,
                                               stream,
                                               mr);
      break;
    default: CUDF_FAIL("binary_operation_safe requires a base-10 decimal output type.");
  }
}

}  // namespace compiled
}  // namespace binops
}  // namespace cudf
