/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/jit/lto/transform_params.cuh>

#include <jit/rtc/cudf.hpp>

#include <chrono>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

// TODO: use this to document how operators can use the accessors
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

TEST_F(RTCTest, CreateFragment)
{
  // TODO: add configuration parameters necessary for testing
  // and profiling cache behaviour
  // TODO: declare getters and setters required for the specific LTO context of the operator
  // they should use the provided LTO functions; might need a planner

  auto fn = []() {
    auto lib = rtc::compile_and_link_udf("test_fragment",
                                         R"***(
    #include "cudf/jit/lto/transform_params.cuh"
    #include "cudf/jit/lto/operators.cuh"
    #include "cudf/jit/lto/scope.cuh"
    #include "cudf/jit/lto/column_view.cuh" // for column_view_core, mutable_column_view_core

    // if we detect that all types are simple types
    // we can exclude some of the getters, setters, and operators
    // for example, we have span, optional_span, and column_view_core

    extern "C" __device__ void transform_operator(cudf::lto::transform_params p){
      using namespace cudf::lto;

      /// <-- BEGIN OF INPUT UNPACKING: Defined by input planner

      // unpack inputs from scope using the appropriate getters based on the LTO context
      using col_user_data = scope::user_data<0>;
      using col0          = scope::column<1, column_view_core, int, false, false>;
      using col1          = scope::column<2, column_view_core, int, false, false>;
      using col2          = scope::column<3, column_view_core, double, false, false>;
      using col3          = scope::column<4, column_view_core, float, false, false>;
      using col4          = scope::column<5, span<float>, float, false, false>;
      using col5          = scope::column<6, optional_span<float>, float, false, true>;
      using out_col       = scope::column<7, mutable_column_view_core, double, false, false>;

      auto user_data = col_user_data::element(p.scope, p.row_index);
      auto in0 = col0::element(p.scope, p.row_index);
      auto in1 = col1::element(p.scope, p.row_index);
      auto in2 = col2::element(p.scope, p.row_index);
      auto in3 = col3::element(p.scope, p.row_index);
      auto in4 = col4::element(p.scope, p.row_index);
      auto in5 = col5::element(p.scope, p.row_index);

      auto result = 0.0;

      /// <-- END OF INPUT UNPACKING


      /// <-- BEGIN OF OPERATOR: Derived from user

      // run operation using the LTO-compiled operators; these should be inlined into the final kernel and optimized together by NVJITLink

      operators::add(&c, &a, &b);
      operators::sub(&c, &a, &b);
      operators::mul(&c, &a, &b);
      operators::mul(&c, &a, &b);
      operators::arctan(&c, &a, &b);
      operators::sqrt(&c, &a, &b);
      operators::cbrt(&c, &a, &b);
      operators::arccos(&c, &a, &b);
      operators::cast_to_float64(&d, &c);

      /// <-- END OF USER-DEFINED OPERATOR


      /// <-- BEGIN OF OUTPUT PACKING: Defined by planner

      // write output to global memory

      out_col::assign(p.scope, p.row_index, result);


      /// <-- END OF OUTPUT PACKING
    }
    )***",
                                         "test_udf_key",
                                         "transform_kernel");

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
