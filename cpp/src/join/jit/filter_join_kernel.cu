/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/row_abi.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <jit/column_accessor.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_JOIN_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

namespace cudf::join::jit {

// TODO: Create a JIT-compatible header for JoinNoMatch to avoid this duplication.
// This must match the definition in cudf/join/join.hpp
constexpr cudf::size_type JoinNoMatch = cuda::std::numeric_limits<cudf::size_type>::min();

template <bool has_user_data, typename... T>
__device__ row_abi::status execute_predicate_op(void* user_data,
                                                size_type row_index,
                                                cuda::std::tuple<T...> args)
{
  return cuda::std::apply(
    [&](auto&&... args) -> row_abi::status {
      if constexpr (has_user_data) {
        if constexpr (cuda::std::is_void_v<decltype(GENERIC_JOIN_FILTER_OP(
                        user_data, row_index, args...))>) {
          GENERIC_JOIN_FILTER_OP(user_data, row_index, args...);
          return 0;
        } else {
          return static_cast<row_abi::status>(
            GENERIC_JOIN_FILTER_OP(user_data, row_index, args...));
        }
      } else {
        if constexpr (cuda::std::is_void_v<decltype(GENERIC_JOIN_FILTER_OP(args...))>) {
          GENERIC_JOIN_FILTER_OP(args...);
          return 0;
        } else {
          return static_cast<row_abi::status>(GENERIC_JOIN_FILTER_OP(args...));
        }
      }
    },
    args);
}

template <bool has_user_data, bool is_null_aware, typename Accessors>
__device__ row_abi::status filter_join_row_operation(cudf::column_device_view_core const* columns,
                                                     row_abi::bool8* outputs,
                                                     void* user_data,
                                                     size_type row)
{
  if constexpr (is_null_aware) {
    cuda::std::optional<bool> result{false};
    auto inputs = Accessors::map(
      [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(columns, row)...}; });
    auto error = execute_predicate_op<has_user_data>(
      user_data, row, cuda::std::tuple_cat(cuda::std::tuple{&result}, inputs));
    if (error != row_abi::success) { return error; }
    *outputs = result.value_or(false);
    return row_abi::success;
  } else {
    auto any_null =
      Accessors::map([&]<typename... A>() { return (A::is_null(columns, row) || ...); });
    if (any_null) {
      *outputs = false;
      return row_abi::success;
    }
    bool result = false;
    auto inputs = Accessors::map(
      [&]<typename... A>() { return cuda::std::tuple{A::element(columns, row)...}; });
    auto error = execute_predicate_op<has_user_data>(
      user_data, row, cuda::std::tuple_cat(cuda::std::tuple{&result}, inputs));
    if (error != row_abi::success) { return error; }
    *outputs = result;
    return row_abi::success;
  }
}
}  // namespace cudf::join::jit

namespace cudf::row_abi {
extern "C" __device__ __inline_hint__ status predicate_operator(void const* inputs,
                                                                bool8* outputs,
                                                                void* user_data,
                                                                i32 row)
{
  return CUDF_KERNEL_INSTANCE(
    static_cast<cudf::column_device_view_core const*>(inputs), outputs, user_data, row);
}
}  // namespace cudf::row_abi
