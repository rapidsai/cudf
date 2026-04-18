

#include "jit/element.cuh"
#include "jit/element_storage.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {

extern "C" __device__ int operation(void* user_data,
                                    int element_index,
                                    void const* const* inputs,
                                    void* const* outputs)
{
  // element_storage tmp0;
  // add(&tmp0, inputs[0], inputs[1]);
  // outputs[0] = tmp0;

  // knows the types
  // cast to the target type
  // load the element
  // perform the operation
  // use C++ AST to codegen
  // store the element
  //
  //
  //
  //
  //
  // pre-compile some of these operators for common operations and types; compiling the operator
  // becomes trivial as we only need to forward declare and then link to the pre-compiled fragments
  //
  // for example, we can pre-compile for all lexicographic comparisons cheaply
  //
  //
  //
  // WE ALSO NEED TO MAKE SURE THE ORDER OF INPUTS AND OUTPUTS doesnt affect the applicability of
  // the generic operators
  //
  //
  //
  // TODO: functions to add fragments
  //
  //
  return 0;
}

// generic_column_operator: catch-all
// specific column_operator

// unary and binary operators can be easily JIT-ed for all types and use JIT-ed operators
//
// the register pressure and compile-time of the generic operator is very high
//
// use PTX to specify the operator to reduce compile time
//
//
//
// should we pre-link some of the kernels to reduce work at JIT-time?

/// @brief The generic transform kernel. Supports all types and nullability combinations.
extern "C" __global__ void compute_columns_kernel(
  size_type row_size,
  bitmask_type const* __restrict__ stencil,
  bool stencil_has_nulls,
  void* __restrict__ user_data,
  column_device_view const* __restrict__ input_cols,
  size_type num_inputs,
  mutable_column_device_view const* __restrict__ output_cols,
  size_type num_outputs,
  bool const* is_scalar)
{
  static constexpr int MAX_INPUTS  = 32;
  static constexpr int MAX_OUTPUTS = 16;

  element_storage input_storage[MAX_INPUTS];
  void const* input_ptrs[MAX_INPUTS];
  element_storage output_storage[MAX_OUTPUTS];
  void* output_ptrs[MAX_OUTPUTS];

  for (int i = 0; i < MAX_INPUTS; ++i) {
    input_ptrs[i] = &input_storage[i];
  }
  for (int i = 0; i < MAX_OUTPUTS; ++i) {
    output_ptrs[i] = &output_storage[i];
  }

  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto element_idx = start; element_idx < row_size; element_idx += stride) {
    auto active_mask = __ballot_sync(0xFFFF'FFFFU, element_idx < row_size);

    for (int i = 0; i < num_inputs; i++) {
      load_element(is_scalar[i], input_cols + i, element_idx, input_storage + i);
    }

    operation(user_data, element_idx, input_ptrs, output_ptrs);

    for (int i = 0; i < num_inputs; i++) {
      store_element(output_cols, output_storage + i, element_idx, active_mask);
    }

    if constexpr (is_null_aware == null_aware::NO) {
      if (stencil_has_nulls && !bit_is_set(stencil, element_idx)) { continue; }

      auto outs = OutputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::output_arg(output_cols, element_idx)...};
      });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto inputs = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, element_idx)...}; });

      if constexpr (has_user_data) {
        auto args =
          cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);

      } else {
        // TODO: static assert invocable
        auto args = cuda::std::tuple_cat(out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);
      }

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, element_idx, cuda::std::get<A::index>(outs)), ...);
      });
    } else {
      bool is_valid[OutputAccessors::size];

      auto outs = OutputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::null_output_arg(output_cols, element_idx)...};
      });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto inputs = InputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::nullable_element(input_cols, element_idx)...};
      });

      if constexpr (has_user_data) {
        auto args =
          cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);

      } else {
        auto args = cuda::std::tuple_cat(out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);
      }

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, element_idx, *cuda::std::get<A::index>(outs)), ...);
        ((is_valid[A::index] = cuda::std::get<A::index>(outs).has_value()), ...);
      });

      OutputAccessors::map([&]<typename... A>() {
        auto active_mask = __ballot_sync(0xFFFF'FFFFU, element_idx < row_size);
        (warp_compact_validity<A>(active_mask, output_cols, element_idx, is_valid[A::index]), ...);
      });
    }
  }
}

}  // namespace cudf