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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <vector>

using entry_type = std::pair<cudf::char_utf8, cudf::char_utf8>;

static void bench_translate(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width   = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width   = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const entry_count = static_cast<cudf::size_type>(state.get_int64("entries"));

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = cudf::strings_column_view(column->view());

  std::vector<entry_type> entries(entry_count);
  std::transform(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(entry_count),
                 entries.begin(),
                 [](auto idx) -> entry_type {
                   return entry_type{'!' + idx, '~' - idx};
                 });

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto chars_size = input.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::translate(input, entries); });
}

NVBENCH_BENCH(bench_translate)
  .set_name("translate")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("entries", {5, 25, 50});
