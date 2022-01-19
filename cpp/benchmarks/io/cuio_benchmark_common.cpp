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

#include <benchmarks/io/cuio_benchmark_common.hpp>

#include <numeric>
#include <string>

#include <unistd.h>

namespace cudf_io = cudf::io;

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
  : type{type}, file_name{random_file_in_dir(tmpdir.path())}
{
}

cudf_io::source_info cuio_source_sink_pair::make_source_info()
{
  switch (type) {
    case io_type::FILEPATH: return cudf_io::source_info(file_name);
    case io_type::HOST_BUFFER: return cudf_io::source_info(buffer.data(), buffer.size());
    default: CUDF_FAIL("invalid input type");
  }
}

cudf_io::sink_info cuio_source_sink_pair::make_sink_info()
{
  switch (type) {
    case io_type::VOID: return cudf_io::sink_info();
    case io_type::FILEPATH: return cudf_io::sink_info(file_name);
    case io_type::HOST_BUFFER: return cudf_io::sink_info(&buffer);
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
