/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_edit_distance_utf8(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  auto input1 = strings_table->get_column(0);
  auto input2 = strings_table->get_column(0);
  auto sv1    = cudf::strings_column_view(input1.view());
  auto sv2    = cudf::strings_column_view(input2.view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.add_global_memory_reads<nvbench::int8_t>(input1.alloc_size() + input2.alloc_size());
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = nvtext::edit_distance(sv1, sv2); });
}

static void bench_edit_distance_ascii(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile profile = data_profile_builder().no_validity().cardinality(0).distribution(
    cudf::type_id::INT32, distribution_id::NORMAL, min_width, max_width);
  data_profile ascii_profile = data_profile_builder().no_validity().cardinality(0).distribution(
    cudf::type_id::INT8, distribution_id::UNIFORM, 32, 126);  // nice ASCII range

  auto offsets = create_random_column(cudf::type_id::INT32, row_count{num_rows + 1}, profile);
  offsets      = cudf::scan(offsets->view(),
                       *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                       cudf::scan_type::EXCLUSIVE);
  auto ascii_data1 =
    create_random_column(cudf::type_id::INT8, row_count{num_rows * max_width}, ascii_profile);
  auto ascii_data2 =
    create_random_column(cudf::type_id::INT8, row_count{num_rows * max_width}, ascii_profile);

  auto input1 = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                  num_rows,
                                  ascii_data1->view().data<char>(),
                                  nullptr,
                                  0,
                                  0,
                                  {offsets->view()});
  auto input2 = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                  num_rows,
                                  ascii_data2->view().data<char>(),
                                  nullptr,
                                  0,
                                  0,
                                  {offsets->view()});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto sv1 = cudf::strings_column_view(input1);
  auto sv2 = cudf::strings_column_view(input2);
  auto chars_size =
    sv1.chars_size(cudf::get_default_stream()) + sv2.chars_size(cudf::get_default_stream());
  auto offsets_size = offsets->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size + 2 * offsets_size);
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = nvtext::edit_distance(sv1, sv2); });
}

static void bench_edit_distance(nvbench::state& state)
{
  auto const encode = state.get_string("encode");
  if (encode == "ascii") {
    bench_edit_distance_ascii(state);
  } else {
    bench_edit_distance_utf8(state);
  }
}

NVBENCH_BENCH(bench_edit_distance)
  .set_name("edit_distance")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {262144, 524288, 1048576})
  .add_string_axis("encode", {"utf8", "ascii"});
