/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
namespace filtering {
namespace jit {

template <bool has_user_data, bool is_null_aware, typename Out, typename... In>
CUDF_KERNEL void kernel(cudf::jit::device_optional_span<typename Out::type> const* outputs,
                        cudf::column_device_view_core const* inputs,
                        void* user_data)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const output = outputs[0].to_span();
  auto const size   = output.size();

  for (auto i = start; i < size; i += stride) {
    bool applies = false;

    if constexpr (!is_null_aware) {
      auto const any_null = (false || ... || In::is_null(inputs, i));

      if (!any_null) {
        if constexpr (has_user_data) {
          GENERIC_FILTER_OP(user_data, i, &applies, In::element(inputs, i)...);
        } else {
          GENERIC_FILTER_OP(&applies, In::element(inputs, i)...);
        }
      }
    } else {
      if constexpr (has_user_data) {
        GENERIC_FILTER_OP(user_data, i, &applies, In::nullable_element(inputs, i)...);
      } else {
        GENERIC_FILTER_OP(&applies, In::nullable_element(inputs, i)...);
      }
    }

    output[i] = applies;
  }
}

}  // namespace jit
}  // namespace filtering
}  // namespace cudf
