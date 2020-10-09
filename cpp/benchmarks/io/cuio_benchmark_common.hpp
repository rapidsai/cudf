/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>

#include <tests/utilities/file_utilities.hpp>

enum io_type { kVoid, kFile, kBuffer };

#define RD_BENCHMARK_DEFINE_ALL_SOURCES(benchmark, name, type_or_group) \
  benchmark(name##_file_input, type_or_group, io_type::kFile);          \
  benchmark(name##_buffer_input, type_or_group, io_type::kBuffer);

#define WR_BENCHMARK_DEFINE_ALL_SINKS(benchmark, name, type_or_group) \
  benchmark(name##_no_output, type_or_group, io_type::kVoid);         \
  benchmark(name##_file_output, type_or_group, io_type::kFile);       \
  benchmark(name##_buffer_output, type_or_group, io_type::kBuffer);

class cuio_source_sink_pair {
 public:
  cuio_source_sink_pair(io_type type);

  ~cuio_source_sink_pair() { std::remove(fname.c_str()); }

  cudf::io::source_info make_source_info();
  cudf::io::sink_info make_sink_info();

 private:
  static temp_directory const tmpdir;

  io_type const type;
  std::vector<char> buffer;
  std::string const fname;
};