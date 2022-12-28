/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <nvbench/nvbench.cuh>

#include <io/json/nested_json.hpp>

#include <tests/io/fst/common.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <cstdlib>

namespace cudf::io::json::detail {
table_with_metadata device_parse_nested_json(
  device_span<SymbolT const> d_input,
  cudf::io::json_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}

namespace cudf {
namespace {

// TODO mix of struct, and list
// Struct, Struct of Struct nested.
// List, List of List nested.
// Or generic any level of nesting.

std::string generate_list_of_lists(int32_t max_depth, int32_t max_rows, std::string elem)
{
  std::string json = "[";
  for (int32_t depth = 1; depth < max_depth; ++depth) {
    json += "[";
  }
  for (int32_t row = 0; row < max_rows; ++row) {
    json += elem;
    if (row < max_rows - 1) { json += ", "; }
  }
  for (int32_t depth = 1; depth < max_depth; ++depth) {
    json += "]";
  }
  json += "]";
  return json;
}

std::string generate_struct_of_structs(int32_t max_depth, int32_t max_rows, std::string elem)
{
  if (max_depth == 0) return "{}";
  std::string json;
  for (int32_t depth = 0; depth < max_depth / 2; ++depth) {
    json += R"({"a)" + std::to_string(depth) + R"(": )";
  }
  if (max_rows == 0) json += "{}";

  for (int32_t row = 0; row < max_rows; ++row) {
    json += elem;
    if (row < max_rows - 1) {
      json += R"(, "a)" + std::to_string(max_depth / 2 - 1) + "_" + std::to_string(row) + R"(": )";
    }
  }
  for (int32_t depth = 0; depth < max_depth / 2; ++depth) {
    json += "}";
  }
  return json;
}

// S20L20S20, elem
// Struct of structs depth 1-127
// Struct of lists depth 1-127
// List of structs depth 1-127
// List of lists depth 1-127
// Mix of List, Struct 1-127

std::string generate_row(
  int num_columns, int max_depth, int max_list_size, int max_struct_size, size_t max_bytes)
{
  std::string s = "{";
  std::vector<std::string> elems{R"(1)", R"(1.1)", R"("1")", R"("abcd")"};
  for (int i = 0; i < num_columns; i++) {
    s += R"("col)" + std::to_string(i) + R"(": )";
    if (max_depth % 2 == 0) {
      s += generate_struct_of_structs(max_depth - 2, (max_struct_size), elems[i % elems.size()]);
    } else {
      s += generate_list_of_lists(max_depth - 2, (max_struct_size), elems[i % elems.size()]);
    }
    if (s.length() > max_bytes) break;
    if (i < num_columns - 1) s += ", ";
  }
  s += "}";
  return s;
}

std::string generate_json(int num_rows,
                          int num_columns,
                          int max_depth,
                          int max_list_size,
                          int max_struct_size,
                          size_t max_json_bytes)
{
  std::string s = "[\n";
  for (int i = 0; i < num_rows; i++) {
    s += generate_row(
      num_columns, max_depth - 2, max_list_size, max_struct_size, max_json_bytes - s.length());
    if (s.length() > max_json_bytes) break;
    if (i != num_rows - 1) s += ",\n";
  }
  s += "\n]";
  // std::cout<<s<<std::endl;
  return s;
}

auto make_test_json_data(size_type string_size, rmm::cuda_stream_view stream)
{
  // Test input
  std::string input = R"(
                      {"a":1,"b":2,"c":[3], "d": {}},
                      {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
                      {"a":1,"b":6.0,"c":[5, 7], "d": null},
                      {"a":1,"b":null,"c":null},
                      {
                        "a" : 1
                      },
                      {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}},
                      {"a": 1, "b": 8.0, "d": { "author": "Jean-Jacques Rousseau"}},)";

  const size_type repeat_times = string_size / input.size();

  auto d_input_scalar   = cudf::make_string_scalar(input, stream);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_input_scalar);
  auto d_scalar         = cudf::strings::repeat_string(d_string_scalar, repeat_times);
  auto& d_input         = static_cast<cudf::scalar_type_t<std::string>&>(*d_scalar);

  auto generated_json    = std::string(d_input);
  generated_json.front() = '[';
  generated_json.back()  = ']';
  return generated_json;
}
}  // namespace

void BM_NESTED_JSON(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const string_size{size_type(state.get_int64("string_size"))};
  auto const default_options = cudf::io::json_reader_options{};

  auto input = make_test_json_data(string_size, cudf::get_default_stream());
  state.add_element_count(input.size());

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::json::detail::device_parse_nested_json(
      input, default_options, cudf::get_default_stream());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_NESTED_JSON)
  .set_name("nested_json_gpu_parser")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

void BM_NESTED_JSON_DEPTH(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const string_size{size_type(state.get_int64("string_size"))};
  auto const depth{size_type(state.get_int64("depth"))};

  auto input = generate_json(100'000'000, 10, depth, 10, 10, string_size);
  // auto input = generate_json(10, 10, depth, 10, 10, string_size);

  state.add_element_count(input.size());
  auto const default_options = cudf::io::json_reader_options{};

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::json::detail::device_parse_nested_json(
      input, default_options, cudf::get_default_stream());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_NESTED_JSON_DEPTH)
  .set_name("nested_json_gpu_parser_depth")
  // .add_int64_axis("depth", nvbench::range(1, 127, 1))
  .add_int64_power_of_two_axis("depth", nvbench::range(1, 6, 1))
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 2));

std::string make_test_json_data_old(size_type string_size, rmm::cuda_stream_view stream);

void BM_NESTED_JSON_DEVICE(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const string_size{size_type(state.get_int64("string_size"))};
  auto default_options = cudf::io::json_reader_options{};
  default_options.enable_lines(true);

  auto input = make_test_json_data_old(string_size, cudf::get_default_stream());
  state.add_element_count(input.size());
  rmm::device_uvector<char> d_input =
    cudf::detail::make_device_uvector_async(input, cudf::get_default_stream());
  device_span<char const> input_span(d_input.data(), d_input.size());

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::json::detail::device_parse_nested_json(
      input_span, default_options, cudf::get_default_stream());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_NESTED_JSON_DEVICE)
  .set_name("nested_json_gpu_parser_device")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

namespace io::detail::json {
cudf::io::table_with_metadata read_json(
  device_span<char const> d_data,
  host_span<char const> h_data,
  cudf::io::json_reader_options const& reader_opts,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}

std::string make_test_json_data_old(size_type string_size, rmm::cuda_stream_view stream)
{
  // Test input
  std::string input = R"(
                      {"a":1,"b":2,"c":"[3]", "d": "{}"}
                      {"a":1,"b":4.0,"c":"[]", "d": "{\"year\":1882,\"author\": \"Bharathi\"}"}
                      {"a":1,"b":6.0,"c":"[5, 7]", "d": null}
                      {"a":2147483647,"b":null,"c":null}
                      { "a" : 1 }
                      {"a":12,"b":Infinity,"c":"[null]", "d": "{\"year\":-600,\"author\": \"Kaniyan\"}"}
                      {"a": -2147483640, "b": 8.0, "d": "{ \"author\": \"Jean-Jacques Rousseau\"}"} )";

  const size_type repeat_times = string_size / input.size();

  auto d_input_scalar   = cudf::make_string_scalar(input, stream);
  auto& d_string_scalar = static_cast<cudf::string_scalar&>(*d_input_scalar);
  auto d_scalar         = cudf::strings::repeat_string(d_string_scalar, repeat_times);
  auto& d_input         = static_cast<cudf::scalar_type_t<std::string>&>(*d_scalar);

  auto generated_json =
    std::string(d_input);  // + "\n{\"d\" : \"" + std::string(4*1024, 'a') + "\"} ";
  generated_json.front() = ' ';
  generated_json.back()  = '\n';
  return generated_json;
}

void BM_OLD_JSON_DEVICE(nvbench::state& state)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const string_size{size_type(state.get_int64("string_size"))};
  auto default_options = cudf::io::json_reader_options{};
  default_options.enable_lines(true);

  auto input = make_test_json_data_old(string_size, cudf::get_default_stream());
  state.add_element_count(input.size());
  rmm::device_uvector<char> d_input =
    cudf::detail::make_device_uvector_async(input, cudf::get_default_stream());
  device_span<char const> input_span(d_input.data(), d_input.size());

  // Run algorithm
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Allocate device-side temporary storage & run algorithm
    cudf::io::detail::json::read_json(
      input_span, input, default_options, cudf::get_default_stream());
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(string_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_OLD_JSON_DEVICE)
  .set_name("old_json_gpu_parser_device")
  .add_int64_power_of_two_axis("string_size", nvbench::range(20, 30, 1));

}  // namespace cudf
