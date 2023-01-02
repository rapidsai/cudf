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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/types.hpp>

#include <cstdlib>

namespace cudf {
namespace {
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

}  // namespace cudf
