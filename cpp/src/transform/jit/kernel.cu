/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/errc.hpp>
#include <cudf/row_abi.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <jit/column_device_view_wrappers.cuh>
#include <jit/transform_kernel.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_TRANSFORM_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

#ifndef CUDF_LTO_MODE
#define CUDF_UDF_TYPE int()
#endif

// Use LTO-dispatch for transform operators if we're in LTO mode. This allows the operator to be
// defined in a separate translation unit and compiled with LTO, which can result in better
// performance due to more optimization opportunities
#ifdef CUDF_LTO_MODE
#define GENERIC_TRANSFORM_OP(...) ::cudf::jit::lto::transform(__VA_ARGS__)
#endif

namespace cudf {
namespace jit {
namespace lto {

using transform_type = CUDF_UDF_TYPE;

extern "C" __device__ transform_type transform;

}  // namespace lto

template <typename... Args>
  requires requires(Args... args) { GENERIC_TRANSFORM_OP(args...); }
__device__ errc invoke_transform(Args... args)
{
  if constexpr (!cuda::std::is_void_v<decltype(GENERIC_TRANSFORM_OP(args...))>) {
    return static_cast<errc>(GENERIC_TRANSFORM_OP(args...));
  } else {
    (void)GENERIC_TRANSFORM_OP(args...);
    return errc::SUCCESS;
  }
}

template <bool has_user_data, typename... Args>
__device__ errc invoke_transform_op(void* user_data, size_type row, Args... args)
{
  if constexpr (has_user_data) {
    return invoke_transform(user_data, row, args...);
  } else {
    return invoke_transform(args...);
  }
}

/// @brief Header-compatible legacy UDF adapter; scheduling lives in the erased kernel.
template <bool is_null_aware, bool has_user_data, typename InputAccessors, typename OutputAccessors>
__device__ errc transform_row_operation(void* user_data,
                                        column_device_view_core const* input_cols,
                                        mutable_column_device_view_core const* output_cols,
                                        size_type row)
{
  if constexpr (is_null_aware) {
    auto outs = OutputAccessors::map(
      [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });
    auto error = OutputAccessors::map([&]<typename... Out>() {
      return InputAccessors::map([&]<typename... In>() {
        return invoke_transform_op<has_user_data>(user_data,
                                                  row,
                                                  &cuda::std::get<Out::index>(outs)...,
                                                  In::nullable_element(input_cols, row)...);
      });
    });
    OutputAccessors::map([&]<typename... A>() {
      (
        [&] {
          auto const& value = cuda::std::get<A::index>(outs);
          if (value.has_value()) { A::assign(output_cols, row, *value); }
          jit::set_validity_warp_cooperative(output_cols[A::index], row, value.has_value());
        }(),
        ...);
    });
    return error;
  } else {
    auto outs = OutputAccessors::map(
      [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });
    auto error = OutputAccessors::map([&]<typename... Out>() {
      return InputAccessors::map([&]<typename... In>() {
        return invoke_transform_op<has_user_data>(
          user_data, row, &cuda::std::get<Out::index>(outs)..., In::element(input_cols, row)...);
      });
    });
    OutputAccessors::map(
      [&]<typename... A>() { (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...); });
    return error;
  }
}
}  // namespace jit
}  // namespace cudf

namespace cudf::row_abi {
extern "C" __device__ __inline_hint__ status transform_operator(void const* inputs,
                                                                void const* outputs,
                                                                void* user_data,
                                                                i32 row)
{
  return static_cast<status>(
    CUDF_KERNEL_INSTANCE(user_data,
                         static_cast<cudf::column_device_view_core const*>(inputs),
                         static_cast<cudf::mutable_column_device_view_core const*>(outputs),
                         row));
}
}  // namespace cudf::row_abi
