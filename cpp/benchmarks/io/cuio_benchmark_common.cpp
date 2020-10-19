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
