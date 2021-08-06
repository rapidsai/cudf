/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.cu
 * @brief cuDF-IO text reader class implementation
 */

#include "cudf/io/text/data_chunk_source.hpp"
#include "cudf/io/text/data_chunk_source_factories.hpp"
#include "cudf/io/text/multibyte_split.hpp"
#include "reader_impl.hpp"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <iterator>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>
#include <array>

namespace cudf {
namespace io {
namespace detail {
namespace text {
// Import functionality that's independent of legacy code
using namespace cudf::io::text;
using namespace cudf::io;

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   text_reader_options const& options,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr), _sources(std::move(sources)), _delimiter(options.get_delimiter())
{
}

table_with_metadata reader::impl::read(std::string delimiter, rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<column>> out_columns;
  std::vector<std::string> delimiters{delimiter};

  // // Iterate through the input sources to acquire data
  // for (auto const& source : _sources) {
  //   if (!source->supports_device_read()) {
  //     // const auto buffer = source->host_read(0, 10000);
  //     // std::string local_buffer { static_cast<char *>(buffer->data()) };
  //     // auto source = cudf::io::text::make_source(local_buffer);
  //     auto source = cudf::io::text::make_source_from_file("/home/jdyer/Desktop/genes.txt");
  //     auto out    = cudf::io::text::multibyte_split(*source, delimiters);
  //     out_columns.push_back(std::move(out));
  //   } else {
  //     //TODO: Read directly to device
  //   }
  // }

  auto source = cudf::io::text::make_source_from_file("/home/jdyer/Desktop/genes.txt");
  auto out    = cudf::io::text::multibyte_split(*source, delimiters);
  printf("Column # of elements: %d\n", out->size());
  out_columns.push_back(std::move(out));

  printf("Out Columns size: %ld\n", out_columns.size());

  return {std::make_unique<table>(std::move(out_columns))};
}

// Forward to implementation
reader::reader(std::vector<std::string> const& filepaths,
               text_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(datasource::create(filepaths), options, mr);
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               text_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(std::move(sources), options, mr);
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(text_reader_options const& options, rmm::cuda_stream_view stream)
{
  return _impl->read(options.get_delimiter(), stream);
}

}  // namespace text
}  // namespace detail
}  // namespace io
}  // namespace cudf
