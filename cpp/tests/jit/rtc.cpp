/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <jit/jit.hpp>

#include <chrono>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

/*
template <int NumInputs,
          int NumOutputs,
          int UserDataIndex,
          typename InputGetters,
          typename OutputSetters>
struct element_operation {
  template <typename Operator>
  static __device__ void evaluate(args scope, cudf::size_type i, Operator&& op)
  {
    if constexpr (UserDataIndex >= 0) {
      auto output_args;
      GENERIC_TRANSFORM_OP(user_data, i, &res, In::element(inputs, i)...);
    } else {
      GENERIC_TRANSFORM_OP(&res, In::element(inputs, i)...);
    }
  }
};
*/

// TODO: cache control policy: ignore cache
// TODO: statistics callback, in cudf layer?
// TODO: extra compile and link flags for LTO library and final kernel, use in CUDF layer, not
// here?
//
// TODO: ??
// TODO: add time function in cache
// TODO: take optional pointer/callback to this for it to be written to
// TODO: take optional list of extra compile flags for linking and compiling
// TODO: cache hit statistics
// global cache statistics before and after?
// TODO: add configuration parameters necessary for testing
// and profiling cache behaviour
// TODO: declare getters and setters required for the specific LTO context of the operator
// they should use the provided LTO functions; might need a planner
// TODO: bincode dump arguments?
// TODO: add a bypass cache argument that forces recompilation for testing purposes
  // TODO: include all headers and disable them based on a macro when not needed for compilation
struct jit_compile_stats {
  std::chrono::nanoseconds cpp_compile_time{};
  std::chrono::nanoseconds fragment_link_time{};
  std::chrono::nanoseconds total_time{};
};

TEST_F(RTCTest, CreateFragmentBasic) {}

TEST_F(RTCTest, CreateFragment)
{
  auto fn = []() {
    char const udf[] = R"***(
    #include "cudf/jit/lto/transform_params.cuh"
    #include "cudf/jit/lto/operators.cuh"
    #include "cudf/jit/lto/scope.cuh"
    #include "cudf/jit/lto/column_view.cuh" // for column_view_core, mutable_column_view_core

    // if we detect that all types are simple types
    // we can exclude some of the getters, setters, and operators
    // for example, we have span, optional_span, and column_view_core

    extern "C" __device__ void transform_operator(cudf::lto::transform_params p){
      using namespace cudf::lto;
      using ops = operators;

      /// <-- BEGIN OF INPUT UNPACKING: Defined by input planner

      // unpack inputs from scope using the appropriate getters based on the LTO context
      using s0          = scope::user_data<0>;
      using s1          = scope::column<1, column_view, int, false, false>;
      using s2          = scope::column<2, column_view, int, false, false>;
      using s3          = scope::column<3, column_view, double, false, false>;
      using s4          = scope::column<4, column_view, float, false, false>;
      using s4          = scope::column<4, string_view, float, false, false>;
      using s4          = scope::column<4, decimal32, float, false, false>;
      using s4          = scope::column<5, column_view, float, false, true>;
      using s4          = scope::column<6, column_view, float, true, false>;
      using s5          = scope::column<7, span<float>, float, false, false>;
      using s6          = scope::column<8, optional_span<float>, float, false, true>;
      using s7          = scope::column<9, mutable_column_view, double, false, false>;

      auto a0 = s0::element(p.scope, p.row_index);
      auto a1 = s1::element(p.scope, p.row_index);
      auto a2 = s2::element(p.scope, p.row_index);
      auto a3 = s3::element(p.scope, p.row_index);
      auto a4 = s4::element(p.scope, p.row_index);
      auto a5 = s5::element(p.scope, p.row_index);
      auto a6 = s6::element(p.scope, p.row_index);
      auto a7 = s7::element(p.scope, p.row_index);
      auto a7 = s8::element(p.scope, p.row_index);
      auto a7 = s9::element(p.scope, p.row_index);

      auto result = 0.0;

      /// <-- END OF INPUT UNPACKING


      /// <-- BEGIN OF OPERATOR: Derived from user

      // run operation using the LTO-compiled operators; these should be inlined into the final kernel and optimized together by NVJITLink

      ops::add(&c, &a, &b);
      ops::sub(&c, &a, &b);
      ops::mul(&c, &a, &b);
      ops::mul(&c, &a, &b);
      ops::arctan(&c, &a, &b);
      ops::sqrt(&c, &a, &b);
      ops::cbrt(&c, &a, &b);
      ops::arccos(&c, &a, &b);
      ops::cast_to_float64(&d, &c);

      /// <-- END OF USER-DEFINED OPERATOR


      /// <-- BEGIN OF OUTPUT PACKING: Defined by planner

      // write output to global memory

      out_col::assign(p.scope, p.row_index, result);


      /// <-- END OF OUTPUT PACKING
    }
    )***";

    auto params = cudf::udf_compile_params{.name                = "test_fragment",
                                           .udf                 = udf,
                                           .key                 = "test_udf_key",
                                           .kernel_symbol       = "transform_kernel",
                                           .extra_compile_flags = {},
                                           .extra_link_flags    = {}};

    auto lib = cudf::compile_and_link_cuda_udf(params);

    auto kernel = lib->get_kernel("transform_kernel");

    EXPECT_EQ("transform_kernel", kernel.get_name());
    // void** scope     = nullptr;
    // int32_t num_rows = 0;
    // void* args[]     = {&scope, &num_rows};
    // kernel.launch(1, 1, 1, 1, 1, 1, 0, nullptr, args);
  };

  fn();
  fn();
  fn();
}

CUDF_TEST_PROGRAM_MAIN()
