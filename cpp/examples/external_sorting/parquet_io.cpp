/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "parquet_io.hpp"
#include "cudf/io/types.hpp"

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <rmm/resource_ref.hpp>

#include <iostream>

namespace cudf {
namespace examples {

std::unique_ptr<cudf::table> read_parquet_file(std::string const& filepath,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::unique_ptr<cudf::table> table;
  std::cout << "Reading: " << filepath << std::endl;
  return cudf::io::read_parquet(cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath)), stream).tbl;
}

}  // namespace examples
}  // namespace cudf
