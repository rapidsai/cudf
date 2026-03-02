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

// TODO: write a planner
// TODO: flags to clear JIT at program startup
// TODO: nvrtc uses the program name to do PCH

TEST_F(RTCTest, CompileKernelBasic)
{
  auto fn = []() {
    char const* udf = R"***(
    #include "cudf/jit/lto/column_view.cuh"
    #include "cudf/jit/lto/operators.cuh"
    #include "cudf/jit/lto/optional_span.cuh"
    #include "cudf/jit/lto/optional.cuh"
    #include "cudf/jit/lto/scope.cuh"
    #include "cudf/jit/lto/span.cuh"
    #include "cudf/jit/lto/string_view.cuh"
    #include "cudf/jit/lto/transform_params.cuh"
    #include "cudf/jit/lto/types.cuh"

    #pragma nv_hdrstop

    extern "C" __device__ void transform_operator(cudf::lto::transform_params p){
      using namespace cudf::lto;

      // unpack inputs from scope using the appropriate getters based on the LTO context
      using s0          = scope::column<0, column_device_view, int, false, false>;
      using s1          = scope::column<1, column_device_view, int, false, false>;
      using s2          = scope::column<2, mutable_column_device_view, int, false, false>;

      auto a0 = s0::element(p.scope, p.row_index);
      auto a1 = s1::element(p.scope, p.row_index);
      int a2;

      operators::add(&a2, &a0, &a1);

      s2::assign(p.scope, p.row_index, a2);
    }
    )***";
    static int i    = 0;

    i++;
    auto key = std::format("test_udf_key_{}", i);

    auto lib = cudf::compile_kernel("test_fragment",
                                    key,
                                    udf,
                                    "transform_kernel",
                                    /*use_cache=*/true,
                                    /*use_pch=*/true,
                                    /*log_pch=*/true);

    auto kernel = lib->get_kernel("transform_kernel");

    EXPECT_EQ("transform_kernel", kernel.get_name());

    auto in0 = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}.release();
    auto in1 = cudf::test::fixed_width_column_wrapper<int>{9, 8, 7, 6, 5, 4, 3, 2, 1}.release();
    auto out = cudf::test::fixed_width_column_wrapper<int>{0, 0, 0, 0, 0, 0, 0, 0, 0}.release();
    int32_t num_rows = 9;

    auto to_device_view = [](auto const& view) {
      std::vector h_view{view};
      rmm::device_uvector<cudf::column_device_view> device_view(1, rmm::cuda_stream_default);
      cudf::detail::cuda_memcpy_async<cudf::column_device_view>(
        device_view, h_view, rmm::cuda_stream_default);
      return device_view;
    };

    auto to_device_mutable_view = [](auto& view) {
      std::vector h_view{view};
      rmm::device_uvector<cudf::mutable_column_device_view> device_view(1,
                                                                        rmm::cuda_stream_default);
      cudf::detail::cuda_memcpy_async<cudf::mutable_column_device_view>(
        device_view, h_view, rmm::cuda_stream_default);
      return device_view;
    };

    auto h_in0     = cudf::column_device_view::create(in0->view());
    auto h_in1     = cudf::column_device_view::create(in1->view());
    auto h_out     = cudf::mutable_column_device_view::create(out->mutable_view());
    auto d_in0     = to_device_view(*h_in0);
    auto d_in1     = to_device_view(*h_in1);
    auto d_out     = to_device_mutable_view(*h_out);
    auto d_in0_ptr = d_in0.data();
    auto d_in1_ptr = d_in1.data();
    auto d_out_ptr = d_out.data();

    rmm::device_buffer d_scope{sizeof(cudf::column_device_view*) +
                                 sizeof(cudf::column_device_view*) +
                                 sizeof(cudf::mutable_column_device_view*),
                               rmm::cuda_stream_default};

    auto* p = static_cast<void**>(d_scope.data());

    detail::cuda_memcpy_async_impl(p,
                                   &d_in0_ptr,
                                   sizeof(cudf::column_device_view*),
                                   detail::host_memory_kind::PAGEABLE,
                                   rmm::cuda_stream_default);
    detail::cuda_memcpy_async_impl(p + 1,
                                   &d_in1_ptr,
                                   sizeof(cudf::column_device_view*),
                                   detail::host_memory_kind::PAGEABLE,
                                   rmm::cuda_stream_default);
    detail::cuda_memcpy_async_impl(p + 2,
                                   &d_out_ptr,
                                   sizeof(cudf::mutable_column_device_view*),
                                   detail::host_memory_kind::PAGEABLE,
                                   rmm::cuda_stream_default);

    auto* scope_arg = d_scope.data();

    void* args[] = {&scope_arg, &num_rows};

    kernel.launch(1, 1, 1, 256, 1, 1, 0, cudaStreamDefault, args);

    auto expected =
      cudf::test::fixed_width_column_wrapper<int>{10, 10, 10, 10, 10, 10, 10, 10, 10}.release();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), expected->view());
  };

  fn();  // warm up cache
  fn();
  fn();
  fn();
}

/*
TEST_F(RTCTest, CreateFragment)
{
  auto fn = []() {
    char const udf[] = R"***(
    #include "cudf/jit/lto/transform_params.cuh"
    #include "cudf/jit/lto/operators.cuh"
    #include "cudf/jit/lto/scope.cuh"
    #include "cudf/jit/lto/column_view.cuh" // for column_device_view, mutable_column_device_view

    // if we detect that all types are simple types
    // we can exclude some of the getters, setters, and operators
    // for example, we have span, optional_span, and column_view_core

    extern "C" __device__ void transform_operator(cudf::lto::transform_params p){
      using namespace cudf::lto;
      using ops = operators;

      /// <-- BEGIN OF INPUT UNPACKING: Defined by input planner

      // unpack inputs from scope using the appropriate getters based on the LTO context
      using s0          = scope::user_data<0>;
      using s1          = scope::column<1, column_device_view, int, false, false>;
      using s2          = scope::column<2, column_device_view, int, false, false>;
      using s3          = scope::column<3, column_device_view, double, false, false>;
      using s4          = scope::column<4, column_device_view, float, false, false>;
      using s4          = scope::column<4, column_device_view, string_view, false, false>;
      using s4          = scope::column<4, column_device_view, decimal32, false, false>;
      using s4          = scope::column<5, column_device_view, float, false, true>;
      using s4          = scope::column<6, column_device_view, float, true, false>;
      using s5          = scope::column<7, span<float>, float, false, false>;
      using s6          = scope::column<8, optional_span<float>, float, false, true>;
      using s7          = scope::column<9, mutable_column_device_view, double, false, false>;

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

      // run operation using the LTO-compiled operators; these should be inlined into the final
kernel and optimized together by NVJITLink

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
*/

CUDF_TEST_PROGRAM_MAIN()
