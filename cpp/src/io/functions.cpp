/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/io/readers.hpp>
#include <cudf/io/writers.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include "orc/chunked_state.hpp"
#include "parquet/chunked_state.hpp"

namespace cudf {
namespace io {
namespace {
template <typename reader, typename reader_options>
std::unique_ptr<reader> make_reader(source_info const& src_info,
                                    reader_options const& options,
                                    rmm::mr::device_memory_resource* mr)
{
  if (src_info.type == io_type::FILEPATH) {
    return std::make_unique<reader>(src_info.filepaths, options, mr);
  }

  std::vector<std::unique_ptr<datasource>> datasources;
  if (src_info.type == io_type::HOST_BUFFER) {
    datasources = cudf::io::datasource::create(src_info.buffers);
  } else if (src_info.type == io_type::USER_IMPLEMENTED) {
    datasources = cudf::io::datasource::create(src_info.user_sources);
  } else {
    CUDF_FAIL("Unsupported source type");
  }

  return std::make_unique<reader>(std::move(datasources), options, mr);
}

template <typename writer, typename writer_options>
std::unique_ptr<writer> make_writer(sink_info const& sink,
                                    writer_options const& options,
                                    rmm::mr::device_memory_resource* mr)
{
  if (sink.type == io_type::FILEPATH) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.filepath), options, mr);
  }
  if (sink.type == io_type::HOST_BUFFER) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.buffer), options, mr);
  }
  if (sink.type == io_type::VOID) {
    return std::make_unique<writer>(cudf::io::data_sink::create(), options, mr);
  }
  if (sink.type == io_type::USER_IMPLEMENTED) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.user_sink), options, mr);
  }
  CUDF_FAIL("Unsupported sink type");
}

}  // namespace

// Freeform API wraps the detail reader class API
table_with_metadata read_avro(read_avro_args const& args, rmm::mr::device_memory_resource* mr)
{
  namespace avro = cudf::io::detail::avro;

  CUDF_FUNC_RANGE();
  avro::reader_options options{args.columns};
  auto reader = make_reader<avro::reader>(args.source, options, mr);

  if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail reader class API
table_with_metadata read_json(read_json_args const& args, rmm::mr::device_memory_resource* mr)
{
  namespace json = cudf::io::detail::json;

  CUDF_FUNC_RANGE();
  json::reader_options options{args.lines, args.compression, args.dtype, args.dayfirst};
  auto reader = make_reader<json::reader>(args.source, options, mr);

  if (args.byte_range_offset != 0 || args.byte_range_size != 0) {
    return reader->read_byte_range(args.byte_range_offset, args.byte_range_size);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail reader class API
table_with_metadata read_csv(read_csv_args const& args, rmm::mr::device_memory_resource* mr)
{
  namespace csv = cudf::io::detail::csv;

  CUDF_FUNC_RANGE();
  csv::reader_options options{};
  options.compression        = args.compression;
  options.lineterminator     = args.lineterminator;
  options.delimiter          = args.delimiter;
  options.decimal            = args.decimal;
  options.thousands          = args.thousands;
  options.comment            = args.comment;
  options.dayfirst           = args.dayfirst;
  options.delim_whitespace   = args.delim_whitespace;
  options.skipinitialspace   = args.skipinitialspace;
  options.skip_blank_lines   = args.skip_blank_lines;
  options.header             = args.header;
  options.infer_date_names   = args.infer_date_names;
  options.infer_date_indexes = args.infer_date_indexes;
  options.names              = args.names;
  options.dtype              = args.dtype;
  options.use_cols_indexes   = args.use_cols_indexes;
  options.use_cols_names     = args.use_cols_names;
  options.true_values.insert(
    options.true_values.end(), args.true_values.begin(), args.true_values.end());
  options.false_values.insert(
    options.false_values.end(), args.false_values.begin(), args.false_values.end());
  if (!args.na_filter) {
    options.na_values.clear();
  } else if (!args.keep_default_na) {
    options.na_values = args.na_values;
  } else {
    options.na_values.insert(options.na_values.end(), args.na_values.begin(), args.na_values.end());
  }
  options.prefix           = args.prefix;
  options.mangle_dupe_cols = args.mangle_dupe_cols;
  options.quotechar        = args.quotechar;
  options.quoting          = args.quoting;
  options.doublequote      = args.doublequote;
  options.timestamp_type   = args.timestamp_type;
  auto reader              = make_reader<csv::reader>(args.source, options, mr);

  if (args.byte_range_offset != 0 || args.byte_range_size != 0) {
    return reader->read_byte_range(args.byte_range_offset, args.byte_range_size);
  } else if (args.skiprows != -1 || args.skipfooter != -1 || args.nrows != -1) {
    return reader->read_rows(args.skiprows, args.skipfooter, args.nrows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail writer class API
void write_csv(write_csv_args const& args, rmm::mr::device_memory_resource* mr)
{
  using namespace cudf::io::detail;

  auto writer = make_writer<csv::writer>(args.sink(), args, mr);

  writer->write_all(args.table(), args.metadata());
}

namespace detail_orc = cudf::io::detail::orc;

// Freeform API wraps the detail reader class API
table_with_metadata read_orc(read_orc_args const& args, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_orc::reader_options options{args.columns,
                                     args.use_index,
                                     args.use_np_dtypes,
                                     args.timestamp_type,
                                     args.decimals_as_float,
                                     args.forced_decimals_scale};
  auto reader = make_reader<detail_orc::reader>(args.source, options, mr);

  if (args.stripe_list.size() > 0) {
    return reader->read_stripes(args.stripe_list);
  } else if (args.stripe != -1) {
    return reader->read_stripe(args.stripe, std::max(args.stripe_count, 1));
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail writer class API
void write_orc(write_orc_args const& args, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_orc::writer_options options{args.compression, args.enable_statistics};
  auto writer = make_writer<detail_orc::writer>(args.sink, options, mr);

  writer->write_all(args.table, args.metadata);
}

/**
 * @copydoc cudf::io::write_orc_chunked_begin
 *
 **/
std::shared_ptr<detail_orc::orc_chunked_state> write_orc_chunked_begin(
  write_orc_chunked_args const& args, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_orc::writer_options options{args.compression, args.enable_statistics};

  auto state = std::make_shared<detail_orc::orc_chunked_state>();
  state->wp  = make_writer<detail_orc::writer>(args.sink, options, mr);

  // have to make a copy of the metadata here since we can't really
  // guarantee the lifetime of the incoming pointer
  if (args.metadata != nullptr) {
    state->user_metadata_with_nullability = *args.metadata;
    state->user_metadata                  = &state->user_metadata_with_nullability;
  }
  state->stream = 0;
  state->wp->write_chunked_begin(*state);
  return state;
}

/**
 * @copydoc cudf::io::write_orc_chunked
 *
 **/
void write_orc_chunked(table_view const& table,
                       std::shared_ptr<detail_orc::orc_chunked_state> state)
{
  CUDF_FUNC_RANGE();
  state->wp->write_chunked(table, *state);
}

/**
 * @copydoc cudf::io::write_orc_chunked_end
 *
 **/
void write_orc_chunked_end(std::shared_ptr<detail_orc::orc_chunked_state>& state)
{
  CUDF_FUNC_RANGE();
  state->wp->write_chunked_end(*state);
  state.reset();
}

using namespace cudf::io::detail::parquet;
namespace detail_parquet = cudf::io::detail::parquet;

// Freeform API wraps the detail reader class API
table_with_metadata read_parquet(read_parquet_args const& args, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_parquet::reader_options options{
    args.columns, args.strings_to_categorical, args.use_pandas_metadata, args.timestamp_type};
  auto reader = make_reader<detail_parquet::reader>(args.source, options, mr);

  if (args.row_groups.size() > 0) {
    return reader->read_row_groups(args.row_groups);
  } else if (args.skip_rows != -1 || args.num_rows != -1) {
    return reader->read_rows(args.skip_rows, args.num_rows);
  } else {
    return reader->read_all();
  }
}

// Freeform API wraps the detail writer class API
std::unique_ptr<std::vector<uint8_t>> write_parquet(write_parquet_args const& args,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_parquet::writer_options options{args.compression, args.stats_level};
  auto writer = make_writer<detail_parquet::writer>(args.sink, options, mr);

  return writer->write_all(
    args.table, args.metadata, args.return_filemetadata, args.metadata_out_file_path);
}

/**
 * @copydoc cudf::io::merge_rowgroup_metadata
 *
 **/
std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list)
{
  CUDF_FUNC_RANGE();
  return detail_parquet::writer::merge_rowgroup_metadata(metadata_list);
}

/**
 * @copydoc cudf::io::write_parquet_chunked_begin
 *
 **/
std::shared_ptr<pq_chunked_state> write_parquet_chunked_begin(
  write_parquet_chunked_args const& args, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  detail_parquet::writer_options options{args.compression, args.stats_level};

  auto state = std::make_shared<pq_chunked_state>();
  state->wp  = make_writer<detail_parquet::writer>(args.sink, options, mr);

  // have to make a copy of the metadata here since we can't really
  // guarantee the lifetime of the incoming pointer
  if (args.metadata != nullptr) {
    state->user_metadata_with_nullability = *args.metadata;
    state->user_metadata                  = &state->user_metadata_with_nullability;
  }
  state->stream = 0;
  state->wp->write_chunked_begin(*state);
  return state;
}

/**
 * @copydoc cudf::io::write_parquet_chunked
 *
 **/
void write_parquet_chunked(table_view const& table, std::shared_ptr<pq_chunked_state> state)
{
  CUDF_FUNC_RANGE();
  state->wp->write_chunked(table, *state);
}

/**
 * @copydoc cudf::io::write_parquet_chunked_end
 *
 **/
std::unique_ptr<std::vector<uint8_t>> write_parquet_chunked_end(
  std::shared_ptr<pq_chunked_state>& state,
  bool return_filemetadata,
  const std::string& metadata_out_file_path)
{
  CUDF_FUNC_RANGE();
  auto meta = state->wp->write_chunked_end(*state, return_filemetadata, metadata_out_file_path);
  state.reset();
  return meta;
}

}  // namespace io
}  // namespace cudf
