/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/io/functions.hpp>
#include <cudf/io/readers.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/mr/device_memory_resource.hpp>

namespace cudf {
namespace experimental {
namespace io {

namespace {

template <typename Reader, typename ReaderOptions>
std::unique_ptr<Reader> make_reader(source_info source, ReaderOptions options) {
  if (source.type == io_type::FILEPATH) {
    return std::make_unique<Reader>(source.filepath, options);
  } else if (source.type == io_type::HOST_BUFFER) {
    return std::make_unique<Reader>(source.buffer.first, source.buffer.second,
                                    options);
  } else if (source.type == io_type::ARROW_RANDOM_ACCESS_FILE) {
    return std::make_unique<Reader>(source.file, options);
  } else {
    CUDF_FAIL("Unsupported source type");
  }
}

}  // namespace

// Freeform API wraps the detail reader class API
table read_avro(read_avro_args const& args,
                rmm::mr::device_memory_resource* mr) {
  namespace avro = cudf::experimental::io::detail::avro;

  avro::reader_options options{args.columns};
  auto reader = make_reader<avro::reader>(args.source, options);

  if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail reader class API
table read_orc(read_orc_args const& args, rmm::mr::device_memory_resource* mr) {
  namespace orc = cudf::experimental::io::detail::orc;

  orc::reader_options options{args.columns, args.use_index, args.use_np_dtypes,
                              args.timestamp_type};
  auto reader = make_reader<orc::reader>(args.source, options);

  if (args.stripe != -1) {
    return reader->read_stripe(args.stripe);
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail reader class API
table read_parquet(read_parquet_args const& args,
                   rmm::mr::device_memory_resource* mr) {
  namespace parquet = cudf::experimental::io::detail::parquet;

  parquet::reader_options options{args.columns, args.strings_to_categorical,
                                  args.use_pandas_metadata,
                                  args.timestamp_type};
  auto reader = make_reader<parquet::reader>(args.source, options);

  if (args.row_group != -1) {
    return reader->read_row_group(args.row_group);
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

}  // namespace io
}  // namespace experimental
}  // namespace cudf
