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

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

namespace {

template <typename NumericType>
std::unique_ptr<cudf::column> get_strings_column(cudf::column_view const& nv)
{
  if constexpr (std::is_floating_point_v<NumericType>) {
    return cudf::strings::from_floats(nv);
  } else {
    return cudf::strings::from_integers(nv);
  }
}
}  // namespace

using Types = nvbench::type_list<float, double, int32_t, int64_t, uint8_t, uint16_t>;

template <typename NumericType>
void bench_convert_number(nvbench::state& state, nvbench::type_list<NumericType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";

  auto const data_type = cudf::data_type(cudf::type_to_id<NumericType>());
  auto const num_col   = create_random_column(data_type.id(), row_count{num_rows});

  auto const strings_col = get_strings_column<NumericType>(num_col->view());
  auto const sv          = cudf::strings_column_view(strings_col->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_num) {
    state.add_global_memory_reads<NumericType>(num_rows);
    state.add_global_memory_writes<int8_t>(sv.chars_size(stream));
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      if constexpr (std::is_floating_point_v<NumericType>) {
        cudf::strings::to_floats(sv, data_type);
      } else {
        cudf::strings::to_integers(sv, data_type);
      }
    });
  } else {
    state.add_global_memory_reads<int8_t>(sv.chars_size(stream));
    state.add_global_memory_writes<NumericType>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      if constexpr (std::is_floating_point_v<NumericType>)
        cudf::strings::from_floats(num_col->view());
      else
        cudf::strings::from_integers(num_col->view());
    });
  }
}

NVBENCH_BENCH_TYPES(bench_convert_number, NVBENCH_TYPE_AXES(Types))
  .set_name("numeric")
  .set_type_axes_names({"NumericType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 16, 1 << 18, 1 << 20, 1 << 22});
