/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <benchmarks/io/cuio_common.hpp>

#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>

#include <unistd.h>

temp_directory const cuio_source_sink_pair::tmpdir{"cudf_gbench"};

std::string random_file_in_dir(std::string const& dir_path)
{
  // `mkstemp` modifies the template in place
  std::string filename = dir_path + "io.XXXXXX";

  // `mkstemp` opens the file; closing immediately, only need the name
  close(mkstemp(const_cast<char*>(filename.data())));

  return filename;
}

cuio_source_sink_pair::cuio_source_sink_pair(io_type type)
  : type{type},
    d_buffer{0, cudf::get_default_stream()},
    file_name{random_file_in_dir(tmpdir.path())}
{
}

cudf::io::source_info cuio_source_sink_pair::make_source_info()
{
  switch (type) {
    case io_type::FILEPATH: return cudf::io::source_info(file_name);
    case io_type::HOST_BUFFER: return cudf::io::source_info(h_buffer.data(), h_buffer.size());
    case io_type::DEVICE_BUFFER: {
      // TODO: make cuio_source_sink_pair stream-friendly and avoid implicit use of the default
      // stream
      auto const stream = cudf::get_default_stream();
      d_buffer.resize(h_buffer.size(), stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_buffer.data(), h_buffer.data(), h_buffer.size(), cudaMemcpyDefault, stream.value()));

      return cudf::io::source_info(d_buffer);
    }
    default: CUDF_FAIL("invalid input type");
  }
}

cudf::io::sink_info cuio_source_sink_pair::make_sink_info()
{
  switch (type) {
    case io_type::VOID: return cudf::io::sink_info(&void_sink);
    case io_type::FILEPATH: return cudf::io::sink_info(file_name);
    case io_type::HOST_BUFFER: [[fallthrough]];
    case io_type::DEVICE_BUFFER: return cudf::io::sink_info(&h_buffer);
    default: CUDF_FAIL("invalid output type");
  }
}

size_t cuio_source_sink_pair::size()
{
  switch (type) {
    case io_type::VOID: return void_sink.bytes_written();
    case io_type::FILEPATH:
      return static_cast<size_t>(
        std::ifstream(file_name, std::ifstream::ate | std::ifstream::binary).tellg());
    case io_type::HOST_BUFFER: [[fallthrough]];
    case io_type::DEVICE_BUFFER: return h_buffer.size();
    default: CUDF_FAIL("invalid output type");
  }
}

std::vector<cudf::type_id> dtypes_for_column_selection(std::vector<cudf::type_id> const& data_types,
                                                       column_selection col_sel)
{
  std::vector<cudf::type_id> out_dtypes;
  out_dtypes.reserve(2 * data_types.size());
  switch (col_sel) {
    case column_selection::ALL:
    case column_selection::FIRST_HALF:
    case column_selection::SECOND_HALF:
      std::copy(data_types.begin(), data_types.end(), std::back_inserter(out_dtypes));
      std::copy(data_types.begin(), data_types.end(), std::back_inserter(out_dtypes));
      break;
    case column_selection::ALTERNATE:
      for (auto const& type : data_types) {
        out_dtypes.push_back(type);
        out_dtypes.push_back(type);
      }
      break;
  }
  return out_dtypes;
}

std::vector<int> select_column_indexes(int num_cols, column_selection col_sel)
{
  std::vector<int> col_idxs(num_cols / 2);
  switch (col_sel) {
    case column_selection::ALL: col_idxs.resize(num_cols);
    case column_selection::FIRST_HALF:
    case column_selection::SECOND_HALF:
      std::iota(std::begin(col_idxs),
                std::end(col_idxs),
                (col_sel == column_selection::SECOND_HALF) ? num_cols / 2 : 0);
      break;
    case column_selection::ALTERNATE:
      for (size_t i = 0; i < col_idxs.size(); ++i)
        col_idxs[i] = 2 * i;
      break;
  }
  return col_idxs;
}

std::vector<std::string> select_column_names(std::vector<std::string> const& col_names,
                                             column_selection col_sel)
{
  auto const col_idxs_to_read = select_column_indexes(col_names.size(), col_sel);

  std::vector<std::string> col_names_to_read;
  std::transform(col_idxs_to_read.begin(),
                 col_idxs_to_read.end(),
                 std::back_inserter(col_names_to_read),
                 [&](auto& idx) { return col_names[idx]; });
  return col_names_to_read;
}

std::vector<cudf::size_type> segments_in_chunk(int num_segments, int num_chunks, int chunk)
{
  CUDF_EXPECTS(num_segments >= num_chunks,
               "Number of chunks cannot be greater than the number of segments in the file");
  auto start_segment = [num_segments, num_chunks](int chunk) {
    return num_segments * chunk / num_chunks;
  };
  std::vector<cudf::size_type> selected_segments;
  for (auto segment = start_segment(chunk); segment < start_segment(chunk + 1); ++segment) {
    selected_segments.push_back(segment);
  }

  return selected_segments;
}

// Executes the command and returns stderr output
std::string exec_cmd(std::string_view cmd)
{
  // Prevent the output from the command from mixing with the original process' output
  std::fflush(nullptr);
  // Switch stderr and stdout to only capture stderr
  auto const redirected_cmd = std::string{"( "}.append(cmd).append(" 3>&2 2>&1 1>&3) 2>/dev/null");
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(redirected_cmd.c_str(), "r"), pclose);
  CUDF_EXPECTS(pipe != nullptr, "popen() failed");

  std::array<char, 128> buffer;
  std::string error_out;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    error_out += buffer.data();
  }
  return error_out;
}

void try_drop_l3_cache()
{
  static bool is_drop_cache_enabled = std::getenv("CUDF_BENCHMARK_DROP_CACHE") != nullptr;
  if (not is_drop_cache_enabled) { return; }

  std::array drop_cache_cmds{"/sbin/sysctl vm.drop_caches=3", "sudo /sbin/sysctl vm.drop_caches=3"};
  CUDF_EXPECTS(std::any_of(drop_cache_cmds.cbegin(),
                           drop_cache_cmds.cend(),
                           [](auto& cmd) { return exec_cmd(cmd).empty(); }),
               "Failed to execute the drop cache command");
}
