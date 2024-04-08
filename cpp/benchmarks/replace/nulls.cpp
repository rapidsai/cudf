/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

static void replace_nulls(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<int32_t>(state.get_int64("row_width"));

  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(max_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_width);

  auto const input_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  auto const input = input_table->view().column(0);
  auto const repl  = input_table->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = cudf::strings_column_view(input).chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);  // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::replace_nulls(input, repl); });
}

NVBENCH_BENCH(replace_nulls)
  .set_name("replace_nulls")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024, 2048})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216});
