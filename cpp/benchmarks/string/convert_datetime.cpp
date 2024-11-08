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

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_D, "cudf::timestamp_D", "cudf::timestamp_D");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_s, "cudf::timestamp_s", "cudf::timestamp_s");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_us, "cudf::timestamp_us", "cudf::timestamp_us");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ns, "cudf::timestamp_ns", "cudf::timestamp_ns");

using Types = nvbench::type_list<cudf::timestamp_D,
                                 cudf::timestamp_s,
                                 cudf::timestamp_ms,
                                 cudf::timestamp_us,
                                 cudf::timestamp_ns>;

template <class DataType>
void bench_convert_datetime(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_ts  = state.get_string("dir") == "from";

  auto const data_type = cudf::data_type(cudf::type_to_id<DataType>());
  auto const ts_col    = create_random_column(data_type.id(), row_count{num_rows});

  auto format = std::string{"%Y-%m-%d %H:%M:%S"};
  auto s_col  = cudf::strings::from_timestamps(ts_col->view(), format);
  auto sv     = cudf::strings_column_view(s_col->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_ts) {
    state.add_global_memory_reads<DataType>(num_rows);
    state.add_global_memory_writes<int8_t>(sv.chars_size(stream));
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::from_timestamps(ts_col->view(), format);
    });
  } else {
    state.add_global_memory_reads<int8_t>(sv.chars_size(stream));
    state.add_global_memory_writes<DataType>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::to_timestamps(sv, data_type, format);
    });
  }
}

NVBENCH_BENCH_TYPES(bench_convert_datetime, NVBENCH_TYPE_AXES(Types))
  .set_name("datetime")
  .set_type_axes_names({"DataType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 16, 1 << 18, 1 << 20, 1 << 22});
