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

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, value_size_min, value_size_max);

  auto const values =
    create_random_column(cudf::type_id::STRING, row_count{num_rows}, table_profile);

  auto delim_scalar  = cudf::make_string_scalar(delim);
  auto delims_column = cudf::make_column_from_scalar(*delim_scalar, num_rows);
  auto input_table   = cudf::table_view({values->view(), delims_column->view()});
  auto input_column  = cudf::strings::concatenate(input_table);

  // extract the chars from the returned strings column.
  auto input_column_contents = input_column->release();
  auto chars_column_contents = input_column_contents.children[1]->release();
  auto chars_buffer          = chars_column_contents.data.release();

  // turn the chars in to a string scalar.
  return cudf::string_scalar(std::move(*chars_buffer));
}

static void BM_multibyte_split(benchmark::State& state)
{
  auto const source_type        = static_cast<data_chunk_source_type>(state.range(0));
  auto const delim_size         = state.range(1);
  auto const delim_percent      = state.range(2);
  auto const file_size_approx   = state.range(3);
  auto const byte_range_percent = state.range(4);

  auto const byte_range_factor = static_cast<double>(byte_range_percent) / 100;
  CUDF_EXPECTS(delim_percent >= 1, "delimiter percent must be at least 1");
  CUDF_EXPECTS(delim_percent <= 50, "delimiter percent must be at most 50");
  CUDF_EXPECTS(byte_range_percent >= 1, "byte range percent must be at least 1");
  CUDF_EXPECTS(byte_range_percent <= 100, "byte range percent must be at most 100");

  auto delim = std::string(delim_size, '0');
  // the algorithm can only support 7 equal characters, so use different chars in the delimiter
  std::iota(delim.begin(), delim.end(), '1');

  auto const delim_factor = static_cast<double>(delim_percent) / 100;
  auto device_input       = create_random_input(file_size_approx, delim_factor, 0.05, delim);
  auto host_input         = thrust::host_vector<char>(device_input.size());
  auto const host_string  = std::string(host_input.data(), host_input.size());

  cudaMemcpyAsync(host_input.data(),
                  device_input.data(),
                  device_input.size() * sizeof(char),
                  cudaMemcpyDeviceToHost,
                  cudf::default_stream_value);

  auto const temp_file_name = random_file_in_dir(temp_dir.path());

  {
    auto temp_fostream = std::ofstream(temp_file_name, std::ofstream::out);
    temp_fostream.write(host_input.data(), host_input.size());
  }

  cudaDeviceSynchronize();

  auto source = [&] {
    switch (source_type) {
      case data_chunk_source_type::file:  //
        return cudf::io::text::make_source_from_file(temp_file_name);
      case data_chunk_source_type::host:  //
        return cudf::io::text::make_source(host_string);
      case data_chunk_source_type::device:  //
        return cudf::io::text::make_source(device_input);
      default: CUDF_FAIL();
    }
  }();

  auto mem_stats_logger   = cudf::memory_stats_logger();
  auto const range_size   = static_cast<int64_t>(device_input.size() * byte_range_factor);
  auto const range_offset = (device_input.size() - range_size) / 2;
  cudf::io::text::byte_range_info range{range_offset, range_size};
  std::unique_ptr<cudf::column> output;
  for (auto _ : state) {
    try_drop_l3_cache();
    cuda_event_timer raii(state, true);
    output = cudf::io::text::multibyte_split(*source, delim, range);
  }
  auto const output_size =
    static_cast<int64_t>(cudf::strings_column_view(output->view()).chars_size());
  state.SetBytesProcessed(state.iterations() * std::max(output_size, range_size));
  state.counters["peak_memory_usage"] = mem_stats_logger.peak_memory_usage();
}

class MultibyteSplitBenchmark : public cudf::benchmark {
};

BENCHMARK_DEFINE_F(MultibyteSplitBenchmark, multibyte_split_simple)(::benchmark::State& state)
{
  BM_multibyte_split(state);
}
BENCHMARK_REGISTER_F(MultibyteSplitBenchmark, multibyte_split_simple)
  ->ArgsProduct(
    {{data_chunk_source_type::device, data_chunk_source_type::file, data_chunk_source_type::host},
     {1, 4, 7},
     {1, 25},
     {1 << 15, 1 << 30},
     {1, 5, 25, 50, 100}})
  ->UseManualTime()
  ->Unit(::benchmark::kMillisecond);
