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

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

table read_avro(avro_read_arg const &args) {
  namespace avro = cudf::io::avro;
  auto reader = [&]() {
    avro::reader_options options{args.columns};

    if (args.source.type == FILE_PATH) {
      return std::make_unique<avro::reader>(args.source.filepath, options);
    } else if (args.source.type == HOST_BUFFER) {
      return std::make_unique<avro::reader>(args.source.buffer.first,
                                           args.source.buffer.second, options);
    } else if (args.source.type == ARROW_RANDOM_ACCESS_FILE) {
      return std::make_unique<avro::reader>(args.source.file, options);
    } else {
      CUDF_FAIL("Unsupported source type");
    }
  }();

  if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

table read_csv(csv_read_arg const &args) {
  namespace csv = cudf::io::csv;
  auto reader = [&]() {
    csv::reader_options options{};

    options.compression = args.compression;
    options.lineterminator = args.lineterminator;
    options.delimiter = args.delimiter;
    options.decimal = args.decimal;
    options.thousands = args.thousands;
    options.comment = args.comment;
    options.dayfirst = args.dayfirst;
    options.delim_whitespace = args.delim_whitespace;
    options.skipinitialspace = args.skipinitialspace;
    options.skip_blank_lines = args.skip_blank_lines;
    options.header = args.header;
    options.names = args.names;
    options.dtype = args.dtype;
    options.use_cols_indexes = args.use_cols_indexes;
    options.use_cols_names = args.use_cols_names;
    options.true_values = args.true_values;
    options.false_values = args.false_values;
    options.na_values = args.na_values;
    options.keep_default_na = args.keep_default_na;
    options.na_filter = args.na_filter;
    options.prefix = args.prefix;
    options.mangle_dupe_cols = args.mangle_dupe_cols;
    options.quotechar = args.quotechar;
    options.quoting = static_cast<csv::quote_style>(args.quoting);
    options.doublequote = args.doublequote;
    options.out_time_unit = args.out_time_unit;

    if (args.source.type == FILE_PATH) {
      return std::make_unique<csv::reader>(args.source.filepath, options);
    } else if (args.source.type == HOST_BUFFER) {
      return std::make_unique<csv::reader>(args.source.buffer.first,
                                           args.source.buffer.second, options);
    } else if (args.source.type == ARROW_RANDOM_ACCESS_FILE) {
      return std::make_unique<csv::reader>(args.source.file, options);
    } else {
      CUDF_FAIL("Unsupported source type");
    }
  }();

  if (args.byte_range_offset != 0 || args.byte_range_size != 0) {
    return reader->read_byte_range(args.byte_range_offset,
                                   args.byte_range_size);
  } else if (args.skiprows != -1 || args.skipfooter != -1 ||
             args.nrows != -1) {
    return reader->read_rows(args.skiprows, args.skipfooter, args.nrows);
  } else {
    return reader->read();
  }
}

table read_json(json_read_arg const &args) {
  namespace json = cudf::io::json;
  CUDF_EXPECTS(args.lines == true, "Only JSONLines are currently supported");

  auto reader = [&]() {
    json::reader_options options{};

    options.source_type = args.source.type;
    if (options.source_type == HOST_BUFFER) {
      options.source =
          std::string(args.source.buffer.first, args.source.buffer.second);
    } else {
      options.source = args.source.filepath;
    }
    options.compression = args.compression;
    options.dtype = args.dtype;
    options.lines = args.lines;

    return std::make_unique<json::reader>(options);
  }();

  if (args.byte_range_offset != 0 || args.byte_range_size != 0) {
    return reader->read_byte_range(args.byte_range_offset,
                                   args.byte_range_size);
  } else {
    return reader->read();
  }
}

table read_orc(orc_read_arg const &args) {
  namespace orc = cudf::io::orc;
  auto reader = [&]() {
    orc::reader_options options{args.columns, args.use_index,
                                args.use_np_dtypes, args.timestamp_unit};

    if (args.source.type == FILE_PATH) {
      return std::make_unique<orc::reader>(args.source.filepath, options);
    } else if (args.source.type == HOST_BUFFER) {
      return std::make_unique<orc::reader>(args.source.buffer.first,
                                           args.source.buffer.second, options);
    } else if (args.source.type == ARROW_RANDOM_ACCESS_FILE) {
      return std::make_unique<orc::reader>(args.source.file, options);
    } else {
      CUDF_FAIL("Unsupported source type");
    }
  }();

  if (args.stripe != -1) {
    return reader->read_stripe(args.stripe);
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

void write_orc(orc_write_arg const &args) {
  namespace orc = cudf::io::orc;
  auto writer = [&]() {
    orc::writer_options options{};

    if (args.sink.type == FILE_PATH) {
      return std::make_unique<orc::writer>(args.sink.filepath, options);
    } else {
      CUDF_FAIL("Unsupported sink type");
    }
  }();

  return writer->write_all(args.table);
}

table read_parquet(parquet_read_arg const &args) {
  namespace parquet = cudf::io::parquet;
  auto reader = [&]() {
    parquet::reader_options options{args.columns, args.strings_to_categorical,
                                    args.use_pandas_metadata,
                                    args.timestamp_unit};

    if (args.source.type == FILE_PATH) {
      return std::make_unique<parquet::reader>(args.source.filepath, options);
    } else if (args.source.type == HOST_BUFFER) {
      return std::make_unique<parquet::reader>(
          args.source.buffer.first, args.source.buffer.second, options);
    } else if (args.source.type == ARROW_RANDOM_ACCESS_FILE) {
      return std::make_unique<parquet::reader>(args.source.file, options);
    } else {
      CUDF_FAIL("Unsupported source type");
    }
  }();

  if (args.row_group != -1) {
    return reader->read_row_group(args.row_group);
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

}  // namespace cudf
