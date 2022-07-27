/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/file_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cstdio>
#include <fstream>
#include <memory>

temp_directory const temp_dir("cudf_gbench");

enum data_chunk_source_type {
  device,
  file,
  host,
};

static cudf::string_scalar create_random_input(int32_t num_chars,
                                               double delim_factor,
                                               double deviation,
                                               std::string delim)
{
  auto const num_delims      = static_cast<int32_t>((num_chars * delim_factor) / delim.size());
  auto const num_delim_chars = num_delims * delim.size();
  auto const num_value_chars = num_chars - num_delim_chars;
  auto const num_rows        = num_delims;
  auto const value_size_avg  = static_cast<int32_t>(num_value_chars / num_rows);
  auto const value_size_min  = static_cast<int32_t>(value_size_avg * (1 - deviation));
  auto const value_size_max  = static_cast<int32_t>(value_size_avg * (1 + deviation));

  data_profile table_profile;

  table_profile.set_distribution_params(  //
    cudf::type_id::STRING,
    distribution_id::NORMAL,
    value_size_min,
    value_size_max);

  auto const values_table = create_random_table(  //
    {cudf::type_id::STRING},
    row_count{num_rows},
    table_profile);

  auto delim_scalar  = cudf::make_string_scalar(delim);
  auto delims_column = cudf::make_column_from_scalar(*delim_scalar, num_rows);
  auto input_table  = cudf::table_view({values_table->get_column(0).view(), delims_column->view()});
  auto input_column = cudf::strings::concatenate(input_table);

  // extract the chars from the returned strings column.
  auto input_column_contents = input_column->release();
  auto chars_column_contents = input_column_contents.children[1]->release();
  auto chars_buffer          = chars_column_contents.data.release();

  // turn the chars in to a string scalar.
  return cudf::string_scalar(std::move(*chars_buffer));
}

static void BM_multibyte_split(benchmark::State& state)
{
  auto source_type      = static_cast<data_chunk_source_type>(state.range(0));
  auto delim_size       = state.range(1);
  auto delim_percent    = state.range(2);
  auto file_size_approx = state.range(3);

  CUDF_EXPECTS(delim_percent >= 1, "delimiter percent must be at least 1");
  CUDF_EXPECTS(delim_percent <= 50, "delimiter percent must be at most 50");

  auto delim = std::string(":", delim_size);

  auto delim_factor = static_cast<double>(delim_percent) / 100;
  auto device_input = create_random_input(file_size_approx, delim_factor, 0.05, delim);
  auto host_input   = thrust::host_vector<char>(device_input.size());
  auto host_string  = std::string(host_input.data(), host_input.size());

  cudaMemcpyAsync(host_input.data(),
                  device_input.data(),
                  device_input.size() * sizeof(char),
                  cudaMemcpyDeviceToHost,
                  cudf::default_stream_value);

  auto temp_file_name = random_file_in_dir(temp_dir.path());

  {
    auto temp_fostream = std::ofstream(temp_file_name, std::ofstream::out);
    temp_fostream.write(host_input.data(), host_input.size());
  }

  cudaDeviceSynchronize();

  auto source = std::unique_ptr<cudf::io::text::data_chunk_source>(nullptr);

  switch (source_type) {
    case data_chunk_source_type::file:  //
      source = cudf::io::text::make_source_from_file(temp_file_name);
      break;
    case data_chunk_source_type::host:  //
      source = cudf::io::text::make_source(host_string);
      break;
    case data_chunk_source_type::device:  //
      source = cudf::io::text::make_source(device_input);
      break;
    default: CUDF_FAIL();
  }

  auto mem_stats_logger = cudf::memory_stats_logger();
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);
    auto output = cudf::io::text::multibyte_split(*source, delim);
  }

  state.SetBytesProcessed(state.iterations() * device_input.size());
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
}

class MultibyteSplitBenchmark : public cudf::benchmark {
};

#define TRANSPOSE_BM_BENCHMARK_DEFINE(name)                                     \
  BENCHMARK_DEFINE_F(MultibyteSplitBenchmark, name)(::benchmark::State & state) \
  {                                                                             \
    BM_multibyte_split(state);                                                  \
  }                                                                             \
  BENCHMARK_REGISTER_F(MultibyteSplitBenchmark, name)                           \
    ->ArgsProduct({{data_chunk_source_type::device,                             \
                    data_chunk_source_type::file,                               \
                    data_chunk_source_type::host},                              \
                   {1, 4, 7},                                                   \
                   {1, 25},                                                     \
                   {1 << 15, 1 << 30}})                                         \
    ->UseManualTime()                                                           \
    ->Unit(::benchmark::kMillisecond);

TRANSPOSE_BM_BENCHMARK_DEFINE(multibyte_split_simple);
