/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>

#include <jit/accessors.cuh>
#include <jit/span.cuh>

// clang-format off
// This header is an inlined header that defines the GENERIC_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace transformation {
namespace jit {

template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void kernel(cudf::mutable_column_device_view_core const* outputs,
                        cudf::column_device_view_core const* inputs,
                        bool* null_mask,
                        void* user_data)
{
  // inputs to JITIFY kernels have to be either sized-integral types or pointers. Structs or
  // references can't be passed directly/correctly as they will be crossing an ABI boundary

  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size   = outputs[0].size();

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      auto const is_valid = (true && ... && In::is_valid(inputs, i));
      if constexpr (may_evaluate_null) { null_mask[i] = is_valid; }

      if (!is_valid) { continue; }

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &Out::element(outputs, i), In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&Out::element(outputs, i), In::element(inputs, i)...);
      }

    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result;

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { null_mask[i] = result.has_value(); }
    }
  }
}

template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void fixed_point_kernel(cudf::mutable_column_device_view_core const* outputs,
                                    cudf::column_device_view_core const* inputs,
                                    bool* null_mask,
                                    void* user_data)
{
  auto const start        = cudf::detail::grid_1d::global_thread_id();
  auto const stride       = cudf::detail::grid_1d::grid_stride();
  auto const size         = outputs[0].size();
  auto const output_scale = static_cast<numeric::scale_type>(outputs[0].type().scale());

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      auto const is_valid = (true && ... && In::is_valid(inputs, i));

      if constexpr (may_evaluate_null) { null_mask[i] = is_valid; }

      if (!is_valid) { continue; }

      typename Out::type result{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}};

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::element(inputs, i)...);
      }

      Out::assign(outputs, i, result);

    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result{
        typename Out::type{numeric::scaled_integer<typename Out::type::rep>{0, output_scale}}};

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { null_mask[i] = result.has_value(); }
    }
  }
}

template <null_aware is_null_aware,
          bool may_evaluate_null,
          bool has_user_data,
          typename Out,
          typename... In>
CUDF_KERNEL void span_kernel(cudf::jit::device_optional_span<typename Out::type> const* outputs,
                             cudf::column_device_view_core const* inputs,
                             bool* null_mask,
                             void* user_data)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size   = outputs[0].size();

  for (auto i = start; i < size; i += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      auto const is_valid = (true && ... && In::is_valid(inputs, i));

      if constexpr (may_evaluate_null) { null_mask[i] = is_valid; }

      if (!is_valid) { continue; }

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &Out::element(outputs, i), In::element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&Out::element(outputs, i), In::element(inputs, i)...);
      }
    } else {  // is_null_aware == null_aware::YES
      cuda::std::optional<typename Out::type> result;

      if constexpr (has_user_data) {
        GENERIC_TRANSFORM_OP(user_data, i, &result, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_TRANSFORM_OP(&result, In::nullable_element(inputs, i)...);
      }

      Out::assign(outputs, i, *result);

      if constexpr (may_evaluate_null) { null_mask[i] = result.has_value(); }
    }
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
