/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/jit/transform_operation.cuh>
#include <cudf/jit/type_tags.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <jit/column_accessor.cuh>
#include <jit/column_device_view_wrappers.cuh>
#include <jit/element_storage.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation-udf.hpp generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
#include <cudf/detail/kernel-instance.hpp>
// clang-format on

namespace cudf {

/// @brief The generic LTO transform kernel. Supports all types and nullability combinations.
/// This kernel is intended to be used with LTO, it has uses registers and has clear memory
/// boundaries. This is intended to be used for simple n-ary operators.
template <bool null_aware, typename InputAccessors, typename OutputAccessors>
__device__ void lto_transform_kernel(
  size_type row_size,
  bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  column_device_view_core const* __restrict__ input_cols,
  mutable_column_device_view_core const* __restrict__ output_cols)
{
  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  static constexpr auto input_layout =
    null_aware ? InputAccessors::map([&]<typename... A>() {
      layout result{};
      ((result = result.unioned(layout_of<typename A::optional_element_type>)), ...);
      return result;
    })
               : InputAccessors::map([&]<typename... A>() {
                   layout result{};
                   ((result = result.unioned(layout_of<typename A::element_type>)), ...);
                   return result;
                 });

  static constexpr auto output_layout =
    null_aware ? OutputAccessors::map([&]<typename... A>() {
      layout result{};
      ((result = result.unioned(layout_of<typename A::optional_element_type>)), ...);
      return result;
    })
               : OutputAccessors::map([&]<typename... A>() {
                   layout result{};
                   ((result = result.unioned(layout_of<typename A::element_type>)), ...);
                   return result;
                 });

  for (auto row = start; row < row_size; row += stride) {
    if constexpr (null_aware) {
      if (stencil != nullptr && !bit_is_set(stencil, row)) { continue; }
    }

    using input_storage_t  = storage<input_layout>;
    using output_storage_t = storage<output_layout>;

    output_storage_t outputs_storage[OutputAccessors::size];
    input_storage_t inputs_storage[InputAccessors::size];

    InputAccessors::map([&]<typename... A>() {
      if constexpr (null_aware) {
        ((*reinterpret_cast<typename A::optional_element_type*>(inputs_storage[A::index].data) =
            A::nullable_element(input_cols, row)),
         ...);
      } else {
        ((*reinterpret_cast<typename A::element_type*>(inputs_storage[A::index].data) =
            A::element(input_cols, row)),
         ...);
      }
    });

    OutputAccessors::map([&]<typename... A>() {
      if constexpr (null_aware) {
        ((*reinterpret_cast<typename A::optional_element_type*>(outputs_storage[A::index].data) =
            A::null_output_arg(output_cols, row)),
         ...);
      } else {
        ((*reinterpret_cast<typename A::element_type*>(outputs_storage[A::index].data) =
            A::output_arg(output_cols, row)),
         ...);
      }
    });

    [[maybe_unused]] auto errc = cudf_transform_operation(user_data,
                                                          row,
                                                          &inputs_storage,
                                                          sizeof(inputs_storage),
                                                          &outputs_storage,
                                                          sizeof(outputs_storage));

    // used only for null-aware
    auto active_mask = null_aware ? __ballot_sync(0xFFFF'FFFFU, row < row_size) : 0xFFFF'FFFFU;

    auto assign = [&]<typename A>() {
      auto* src = outputs_storage[A::index].data;
      if constexpr (null_aware) {
        auto& ret = *reinterpret_cast<typename A::optional_element_type*>(src);
        A::assign(output_cols, row, *ret);
        jit::warp_compact_validity<A>(active_mask, output_cols, row, ret.has_value());
      } else {
        A::assign(output_cols, row, *reinterpret_cast<typename A::element_type*>(src));
      }
    };

    OutputAccessors::map([&]<typename... A>() { (assign.template operator()<A>(), ...); });
  }
}

}  // namespace cudf

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type row_size,
  cudf::bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  cudf::column_device_view_core const* __restrict__ input_cols,
  cudf::mutable_column_device_view_core const* __restrict__ output_cols)
{
  CUDF_KERNEL_INSTANCE(row_size, stencil, user_data, input_cols, output_cols);
}
