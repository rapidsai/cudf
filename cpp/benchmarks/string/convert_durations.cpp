/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/durations.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_D, "cudf::duration_D", "cudf::duration_D");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_s, "cudf::duration_s", "cudf::duration_s");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_ms, "cudf::duration_ms", "cudf::duration_ms");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_us, "cudf::duration_us", "cudf::duration_us");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_ns, "cudf::duration_ns", "cudf::duration_ns");

using Types = nvbench::type_list<cudf::duration_D,
                                 cudf::duration_s,
                                 cudf::duration_ms,
                                 cudf::duration_us,
                                 cudf::duration_ns>;

template <class DataType>
void bench_convert_duration(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type = cudf::data_type(cudf::type_to_id<DataType>());
  auto const from_dur  = state.get_string("dir") == "from";

  auto const ts_col = create_random_column(data_type.id(), row_count{num_rows});
  cudf::column_view input(ts_col->view());

  auto format = std::string{"%D days %H:%M:%S"};
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_dur) {
    state.add_global_memory_reads<DataType>(num_rows);
    state.add_global_memory_writes<int8_t>(format.size() * num_rows);
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::from_durations(input, format); });
  } else {
    auto source = cudf::strings::from_durations(input, format);
    auto view   = cudf::strings_column_view(source->view());
    state.add_global_memory_reads<int8_t>(view.chars_size(stream));
    state.add_global_memory_writes<DataType>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::to_durations(view, data_type, format);
    });
  }
}

NVBENCH_BENCH_TYPES(bench_convert_duration, NVBENCH_TYPE_AXES(Types))
  .set_name("duration")
  .set_type_axes_names({"DataType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 10, 1 << 15, 1 << 20, 1 << 25});
