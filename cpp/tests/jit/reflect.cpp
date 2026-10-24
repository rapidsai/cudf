/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>

#include <cudf_fragments.hpp>
#include <gtest/gtest.h>
#include <jit/reflect.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string compact(std::string value)
{
  std::erase_if(value, [](unsigned char c) { return std::isspace(c); });
  return value;
}

using cudf::lto_binary_type;
using cudf::transform_input_spec;
using cudf::transform_output_spec;
using cudf::type_id;
using cudf::udf_source_type;

TEST(JitReflectionTest, PrecompiledFragmentsMatchRuntimeReflection)
{
  auto const expect_precompiled = [](auto const& inputs, auto const& outputs) {
    auto const reflection = cudf::jit::reflect(lto_binary_type::LTO_IR, inputs, outputs);
    auto const name       = compact(std::format("cudf::jit::transform_kernel<false, false, {}, {}>",
                                          reflection.input_accessors,
                                          reflection.output_accessors));
    EXPECT_TRUE(std::any_of(std::begin(cudf_fragments::transform_kernel_INSTANCE),
                            std::end(cudf_fragments::transform_kernel_INSTANCE),
                            [&](auto const* instance) { return compact(instance) == name; }))
      << "Missing precompiled fragment: " << name;
  };

  for (auto const type : {type_id::UINT8,
                          type_id::UINT16,
                          type_id::UINT32,
                          type_id::UINT64,
                          type_id::DECIMAL32,
                          type_id::DECIMAL64,
                          type_id::DECIMAL128}) {
    std::array const outputs{transform_output_spec{.type = type}};
    std::array const unary_inputs{transform_input_spec{.type = type}};
    expect_precompiled(unary_inputs, outputs);
    for (auto const scalar : {false, true}) {
      std::array const binary_inputs{transform_input_spec{.type = type},
                                     transform_input_spec{.type = type, .is_scalar = scalar}};
      expect_precompiled(binary_inputs, outputs);
    }
  }

  std::array const inputs{transform_input_spec{.type = type_id::STRING}};
  for (auto const type :
       {type_id::UINT8, type_id::UINT16, type_id::UINT32, type_id::UINT64, type_id::STRING}) {
    for (int count = 1; count <= 8; ++count) {
      std::vector<transform_output_spec> const outputs(
        count, {.type = type, .has_string_offsets = type == type_id::STRING});
      expect_precompiled(inputs, outputs);
    }
  }
}

TEST(JitReflectionTest, DefaultAndExplicitTableMappings)
{
  std::array const inputs{transform_input_spec{.type = type_id::FLOAT32},
                          transform_input_spec{.type = type_id::INT16, .is_scalar = true}};
  std::array const outputs{transform_output_spec{.type = type_id::STRING}};
  std::array<int32_t, 2> const zero_sources{0, 0}, default_indices{0, 1};
  std::array<int32_t, 1> const zero_output{0};
  auto const defaults = cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs);
  EXPECT_EQ(defaults,
            cudf::jit::reflect(udf_source_type::CUDA,
                               inputs,
                               outputs,
                               zero_sources,
                               default_indices,
                               zero_output,
                               zero_output));

  std::array<int32_t, 2> const sources{3, 2}, indices{7, 4};
  std::array<int32_t, 1> const output_sources{5}, output_indices{8};
  auto const mapped = cudf::jit::reflect(
    udf_source_type::CUDA, inputs, outputs, sources, indices, output_sources, output_indices);
  EXPECT_EQ(compact(mapped.input_accessors),
            "cudf::jit::type_list<cudf::jit::column_accessor<7,float,false,3>,"
            "cudf::jit::column_accessor<4,int16_t,true,2>>");
  EXPECT_EQ(compact(mapped.output_accessors),
            "cudf::jit::type_list<cudf::jit::column_accessor<8,"
            "cudf::repr_element<cudf::string_view>,false,5>>");
  EXPECT_NE(defaults, mapped);
}

TEST(JitReflectionTest, InvalidTableMappings)
{
  std::array const inputs{transform_input_spec{.type = type_id::INT32}};
  std::array const outputs{transform_output_spec{.type = type_id::INT32}};
  std::array<int32_t, 1> const one{0};
  std::array<int32_t, 2> const two{0, 1};
  EXPECT_THROW(cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs, two, two),
               std::runtime_error);
  EXPECT_THROW(cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs, {}, one),
               std::runtime_error);
  EXPECT_THROW(cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs, {}, {}, two, two),
               std::runtime_error);
  EXPECT_THROW(cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs, {}, {}, one, {}),
               std::runtime_error);
}

TEST(JitReflectionTest, PhysicalTypesAndPtxMetadata)
{
  std::array const inputs{transform_input_spec{.type = type_id::FLOAT32}};
  std::array const outputs{transform_output_spec{.type = type_id::FLOAT64}};
  auto const cuda = cudf::jit::reflect(udf_source_type::CUDA, inputs, outputs);
  auto const ptx  = cudf::jit::reflect(udf_source_type::PTX, inputs, outputs);
  EXPECT_EQ(cuda.input_accessors, ptx.input_accessors);
  EXPECT_EQ(cuda.output_accessors, ptx.output_accessors);
  EXPECT_TRUE(cuda.ptx_input_value_types.empty());
  EXPECT_TRUE(cuda.ptx_output_value_types.empty());
  EXPECT_EQ(ptx.ptx_input_value_types, std::vector<std::string>{"float"});
  EXPECT_EQ(ptx.ptx_output_value_types, std::vector<std::string>{"double"});
  EXPECT_NE(cuda, ptx);
  for (auto const format :
       {lto_binary_type::LTO_IR, lto_binary_type::FATBIN, lto_binary_type::PTX}) {
    auto const lto = cudf::jit::reflect(format, inputs, outputs);
    EXPECT_EQ(compact(lto.input_accessors),
              "cudf::jit::type_list<cudf::jit::column_accessor<0,uint32_t,false,0>>");
    EXPECT_EQ(compact(lto.output_accessors),
              "cudf::jit::type_list<cudf::jit::column_accessor<0,uint64_t,false,0>>");
    EXPECT_TRUE(lto.ptx_input_value_types.empty());
    EXPECT_TRUE(lto.ptx_output_value_types.empty());
  }
}

TEST(JitReflectionTest, UdfSignaturesUseValuesRatherThanAccessorTags)
{
  std::array const inputs{
    transform_input_spec{.type     = type_id::DICTIONARY32,
                         .children = {{.type = type_id::INT32}, {.type = type_id::STRING}}}};
  std::array const outputs{
    transform_output_spec{.type = type_id::STRING},
    transform_output_spec{.type = type_id::STRING, .has_string_offsets = true}};
  EXPECT_EQ(compact(cudf::jit::reflect_udf_signature(false, false, inputs, outputs, false)),
            "int(cudf::string_view*,cudf::mutable_string_view*,cudf::string_view)");
  EXPECT_EQ(compact(cudf::jit::reflect_udf_signature(true, true, inputs, outputs, true)),
            "int(void*,cudf::size_type,cuda::std::optional<cudf::string_view>*,"
            "cuda::std::optional<cudf::mutable_string_view>*,"
            "cuda::std::optional<cudf::string_view>)");
  auto const reflection = cudf::jit::reflect(udf_source_type::PTX, inputs, outputs);
  EXPECT_EQ(reflection.ptx_input_value_types, std::vector<std::string>{"cudf::string_view"});
  EXPECT_EQ(reflection.ptx_output_value_types,
            (std::vector<std::string>{"cudf::string_view", "cudf::mutable_string_view"}));
}

TEST(JitReflectionTest, ColumnStorageAndSpecificationsAgree)
{
  cudf::test::fixed_width_column_wrapper<float> input{1.0f, 2.0f};
  std::array<cudf::transform_input, 1> const inputs{input};
  std::array<cudf::jit::output_column, 1> const outputs{
    cudf::jit::fixed_width_column{cudf::make_numeric_column(cudf::data_type{type_id::FLOAT64}, 2)}};
  auto const input_specs  = cudf::jit::make_input_specs(inputs);
  auto const output_specs = cudf::jit::make_output_specs(outputs);
  std::array<int32_t, 1> const sources{2}, indices{5};
  for (std::variant<udf_source_type, lto_binary_type> const source :
       {std::variant<udf_source_type, lto_binary_type>{udf_source_type::CUDA},
        std::variant<udf_source_type, lto_binary_type>{udf_source_type::PTX},
        std::variant<udf_source_type, lto_binary_type>{lto_binary_type::LTO_IR}}) {
    EXPECT_EQ(
      cudf::jit::reflect(source, inputs, outputs, sources, indices, sources, indices),
      cudf::jit::reflect(source, input_specs, output_specs, sources, indices, sources, indices));
  }
}

}  // namespace

CUDF_TEST_PROGRAM_MAIN()
