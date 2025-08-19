/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <concepts>
#include <vector>

template <typename T>
struct benchmark_data;

template <std::floating_point T>
struct benchmark_data<T> {
  static T dist_min() { return 0; }

  static T dist_max() { return 1; }

  static T filter_min() { return 0.05; }

  static T filter_max() { return 0.07; }
};

template <std::integral T>
struct benchmark_data<T> {
  static T dist_min() { return -128; }

  static T dist_max() { return 127; }

  static T filter_min() { return -64; }

  static T filter_max() { return 64; }
};

enum class engine_type : uint8_t { AST = 0, JIT = 1 };

engine_type engine_from_string(std::string_view str)
{
  if (str == "ast") {
    return engine_type::AST;
  } else if (str == "jit") {
    return engine_type::JIT;
  } else {
    CUDF_FAIL("unrecognized engine enum: " + std::string(str));
  }
}

bool boolean_from_string(std::string_view str)
{
  if (str == "true") {
    return true;
  } else if (str == "false") {
    return false;
  } else {
    CUDF_FAIL("unrecognized boolean value: " + std::string(str));
  }
}

template <typename key_type>
static void BM_filter_min_max(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const engine_name = state.get_string("engine");
  auto const nullable    = boolean_from_string(state.get_string("nullable"));
  auto const engine      = engine_from_string(engine_name);

  auto profile = data_profile{};
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  benchmark_data<key_type>::dist_min(),
                                  benchmark_data<key_type>::dist_max());
  profile.set_null_probability(nullable ? std::optional{0.3} : std::nullopt);

  auto const column =
    create_random_column(cudf::type_to_id<key_type>(), row_count{num_rows}, profile);

  std::string type_name = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});

  auto udf = std::format(
    R"***(
    __device__ void transform(bool * out, {0} c0, {0} min, {0} max) {{
    *out = (c0 >= min && c0 <= max);
    }}
    )***",
    type_name);

  auto tree              = cudf::ast::tree{};
  auto min_scalar        = cudf::numeric_scalar<key_type>{benchmark_data<key_type>::filter_min()};
  auto max_scalar        = cudf::numeric_scalar<key_type>{benchmark_data<key_type>::filter_max()};
  auto min_scalar_column = cudf::make_column_from_scalar(min_scalar, 1);
  auto max_scalar_column = cudf::make_column_from_scalar(max_scalar, 1);

  {
    auto& column_ref  = tree.push(cudf::ast::column_reference{0});
    auto& min_literal = tree.push(cudf::ast::literal{min_scalar});
    auto& max_literal = tree.push(cudf::ast::literal{max_scalar});
    auto& filter_min  = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, column_ref, min_literal});
    auto& filter_max =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::LESS_EQUAL, column_ref, max_literal});
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::LOGICAL_AND, filter_min, filter_max});
  }

  std::vector<cudf::column_view> filter_inputs;
  filter_inputs.push_back(column->view());
  filter_inputs.push_back(min_scalar_column->view());
  filter_inputs.push_back(max_scalar_column->view());

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(static_cast<size_t>(num_rows));
  state.add_global_memory_writes<key_type>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto stream = launch.get_stream().get_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    switch (engine) {
      case engine_type::AST: {
        auto input_table          = cudf::table_view{{column->view()}};
        auto const filter_boolean = cudf::compute_column(input_table, tree.back(), stream, mr);
        auto const result =
          cudf::apply_boolean_mask(input_table, filter_boolean->view(), stream, mr);
      } break;
      case engine_type::JIT: {
        auto result = cudf::filter(filter_inputs,
                                   udf,
                                   false,
                                   std::nullopt,
                                   std::vector{true, false, false},
                                   cudf::null_aware::NO,
                                   stream,
                                   mr);
      } break;
      default: CUDF_UNREACHABLE("Unrecognised engine type requested");
    }
  });
}

#define FILTER_BENCHMARK_DEFINE(name, key_type)                                 \
  static void name(::nvbench::state& st) { ::BM_filter_min_max<key_type>(st); } \
  NVBENCH_BENCH(name)                                                           \
    .set_name(#name)                                                            \
    .add_string_axis("engine", {"ast", "jit"})                                  \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})  \
    .add_string_axis("nullable", {"true", "false"})

FILTER_BENCHMARK_DEFINE(filter_min_max_int32, int32_t);
FILTER_BENCHMARK_DEFINE(filter_min_max_int64, int64_t);
FILTER_BENCHMARK_DEFINE(filter_min_max_float32, float);
FILTER_BENCHMARK_DEFINE(filter_min_max_float64, double);
