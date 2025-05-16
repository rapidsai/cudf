/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

using Types = nvbench::type_list<numeric::decimal32, numeric::decimal64>;

NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal32, "decimal32", "decimal32");
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal64, "decimal64", "decimal64");

template <typename DataType>
void bench_convert_fixed_point(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";

  auto const data_type = cudf::data_type{cudf::type_to_id<DataType>(), numeric::scale_type{-2}};
  auto const fp_col    = create_random_column(data_type.id(), row_count{num_rows});

  auto const strings_col = cudf::strings::from_fixed_point(fp_col->view());
  auto const sv          = cudf::strings_column_view(strings_col->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_num) {
    state.add_global_memory_reads<int8_t>(num_rows * cudf::size_of(data_type));
    state.add_global_memory_writes<int8_t>(sv.chars_size(stream));
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::to_fixed_point(sv, data_type); });
  } else {
    state.add_global_memory_reads<int8_t>(sv.chars_size(stream));
    state.add_global_memory_writes<int8_t>(num_rows * cudf::size_of(data_type));
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::from_fixed_point(fp_col->view()); });
  }
}

NVBENCH_BENCH_TYPES(bench_convert_fixed_point, NVBENCH_TYPE_AXES(Types))
  .set_name("fixed_point")
  .set_type_axes_names({"DataType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 16, 1 << 18, 1 << 20, 1 << 22});
