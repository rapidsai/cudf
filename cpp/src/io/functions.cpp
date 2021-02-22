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

#include <io/orc/orc.h>
#include <algorithm>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/avro.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {
// Returns builder for csv_reader_options
csv_reader_options_builder csv_reader_options::builder(source_info const& src)
{
  return csv_reader_options_builder{src};
}

// Returns builder for csv_writer_options
csv_writer_options_builder csv_writer_options::builder(sink_info const& sink,
                                                       table_view const& table)
{
  return csv_writer_options_builder{sink, table};
}

// Returns builder for orc_reader_options
orc_reader_options_builder orc_reader_options::builder(source_info const& src)
{
  return orc_reader_options_builder{src};
}

// Returns builder for orc_writer_options
orc_writer_options_builder orc_writer_options::builder(sink_info const& sink,
                                                       table_view const& table)
{
  return orc_writer_options_builder{sink, table};
}

// Returns builder for chunked_orc_writer_options
chunked_orc_writer_options_builder chunked_orc_writer_options::builder(sink_info const& sink)
{
  return chunked_orc_writer_options_builder{sink};
}

// Returns builder for avro_reader_options
avro_reader_options_builder avro_reader_options::builder(source_info const& src)
{
  return avro_reader_options_builder(src);
}

// Returns builder for json_reader_options
json_reader_options_builder json_reader_options::builder(source_info const& src)
{
  return json_reader_options_builder(src);
}

// Returns builder for parquet_reader_options
parquet_reader_options_builder parquet_reader_options::builder(source_info const& src)
{
  return parquet_reader_options_builder{src};
}

// Returns builder for parquet_writer_options
parquet_writer_options_builder parquet_writer_options::builder(sink_info const& sink,
                                                               table_view const& table)
{
  return parquet_writer_options_builder{sink, table};
}

// Returns builder for parquet_writer_options
parquet_writer_options_builder parquet_writer_options::builder()
{
  return parquet_writer_options_builder();
}

// Returns builder for chunked_parquet_writer_options
chunked_parquet_writer_options_builder chunked_parquet_writer_options::builder(
  sink_info const& sink)
{
  return chunked_parquet_writer_options_builder(sink);
}

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

template <typename writer, typename... Ts>
std::unique_ptr<writer> make_writer(sink_info const& sink, Ts&&... args)
{
  if (sink.type == io_type::FILEPATH) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.filepath),
                                    std::forward<Ts>(args)...);
  }
  if (sink.type == io_type::HOST_BUFFER) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.buffer),
                                    std::forward<Ts>(args)...);
  }
  if (sink.type == io_type::VOID) {
    return std::make_unique<writer>(cudf::io::data_sink::create(), std::forward<Ts>(args)...);
  }
  if (sink.type == io_type::USER_IMPLEMENTED) {
    return std::make_unique<writer>(cudf::io::data_sink::create(sink.user_sink),
                                    std::forward<Ts>(args)...);
  }
  CUDF_FAIL("Unsupported sink type");
}

}  // namespace

table_with_metadata read_avro(avro_reader_options const& opts, rmm::mr::device_memory_resource* mr)
{
  namespace avro = cudf::io::detail::avro;

  CUDF_FUNC_RANGE();
  auto reader = make_reader<avro::reader>(opts.get_source(), opts, mr);
  return reader->read(opts);
}

table_with_metadata read_json(json_reader_options const& opts, rmm::mr::device_memory_resource* mr)
{
  namespace json = cudf::io::detail::json;

  CUDF_FUNC_RANGE();
  auto reader = make_reader<json::reader>(opts.get_source(), opts, mr);
  return reader->read(opts);
}

table_with_metadata read_csv(csv_reader_options const& options, rmm::mr::device_memory_resource* mr)
{
  namespace csv = cudf::io::detail::csv;

  CUDF_FUNC_RANGE();
  auto reader = make_reader<csv::reader>(options.get_source(), options, mr);

  return reader->read();
}

// Freeform API wraps the detail writer class API
void write_csv(csv_writer_options const& options, rmm::mr::device_memory_resource* mr)
{
  using namespace cudf::io::detail;

  auto writer = make_writer<csv::writer>(options.get_sink(), options, mr);

  writer->write(options.get_table(), options.get_metadata());
}

namespace detail_orc = cudf::io::detail::orc;

raw_orc_statistics read_raw_orc_statistics(source_info const& src_info)
{
  // Get source to read statistics from
  std::unique_ptr<datasource> source;
  if (src_info.type == io_type::FILEPATH) {
    CUDF_EXPECTS(src_info.filepaths.size() == 1, "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.filepaths[0]);
  } else if (src_info.type == io_type::HOST_BUFFER) {
    CUDF_EXPECTS(src_info.buffers.size() == 1, "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.buffers[0]);
  } else if (src_info.type == io_type::USER_IMPLEMENTED) {
    CUDF_EXPECTS(src_info.user_sources.size() == 1, "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.user_sources[0]);
  } else {
    CUDF_FAIL("Unsupported source type");
  }

  orc::metadata metadata(source.get());

  // Initialize statistics to return
  raw_orc_statistics result;

  // Get column names
  for (auto i = 0; i < metadata.get_num_columns(); i++) {
    result.column_names.push_back(metadata.get_column_name(i));
  }

  // Get file-level statistics, statistics of each column of file
  for (auto const& stats : metadata.ff.statistics) {
    result.file_stats.push_back(std::string(stats.cbegin(), stats.cend()));
  }

  // Get stripe-level statistics
  for (auto const& stripes_stats : metadata.md.stripeStats) {
    result.stripes_stats.emplace_back();
    for (auto const& stats : stripes_stats.colStats) {
      result.stripes_stats.back().push_back(std::string(stats.cbegin(), stats.cend()));
    }
  }

  return result;
}

column_statistics::column_statistics(cudf::io::orc::column_statistics&& cs)
{
  _number_of_values = std::move(cs.number_of_values);
  if (cs.int_stats.get()) {
    _type                = statistics_type::INT;
    _type_specific_stats = cs.int_stats.release();
  } else if (cs.double_stats.get()) {
    _type                = statistics_type::DOUBLE;
    _type_specific_stats = cs.double_stats.release();
  } else if (cs.string_stats.get()) {
    _type                = statistics_type::STRING;
    _type_specific_stats = cs.string_stats.release();
  } else if (cs.bucket_stats.get()) {
    _type                = statistics_type::BUCKET;
    _type_specific_stats = cs.bucket_stats.release();
  } else if (cs.decimal_stats.get()) {
    _type                = statistics_type::DECIMAL;
    _type_specific_stats = cs.decimal_stats.release();
  } else if (cs.date_stats.get()) {
    _type                = statistics_type::DATE;
    _type_specific_stats = cs.date_stats.release();
  } else if (cs.binary_stats.get()) {
    _type                = statistics_type::BINARY;
    _type_specific_stats = cs.binary_stats.release();
  } else if (cs.timestamp_stats.get()) {
    _type                = statistics_type::TIMESTAMP;
    _type_specific_stats = cs.timestamp_stats.release();
  }
}

column_statistics& column_statistics::operator=(column_statistics&& other) noexcept
{
  _number_of_values    = std::move(other._number_of_values);
  _type                = other._type;
  _type_specific_stats = other._type_specific_stats;

  other._type                = statistics_type::NONE;
  other._type_specific_stats = nullptr;

  return *this;
}

column_statistics::column_statistics(column_statistics&& other) noexcept
{
  *this = std::move(other);
}

column_statistics::~column_statistics()
{
  switch (_type) {
    case statistics_type::NONE:  // error state, but can't throw from a destructor.
      break;
    case statistics_type::INT: delete static_cast<integer_statistics*>(_type_specific_stats); break;
    case statistics_type::DOUBLE:
      delete static_cast<double_statistics*>(_type_specific_stats);
      break;
    case statistics_type::STRING:
      delete static_cast<string_statistics*>(_type_specific_stats);
      break;
    case statistics_type::BUCKET:
      delete static_cast<bucket_statistics*>(_type_specific_stats);
      break;
    case statistics_type::DECIMAL:
      delete static_cast<decimal_statistics*>(_type_specific_stats);
      break;
    case statistics_type::DATE: delete static_cast<date_statistics*>(_type_specific_stats); break;
    case statistics_type::BINARY:
      delete static_cast<binary_statistics*>(_type_specific_stats);
      break;
    case statistics_type::TIMESTAMP:
      delete static_cast<timestamp_statistics*>(_type_specific_stats);
      break;
  }
}

parsed_orc_statistics read_parsed_orc_statistics(source_info const& src_info)
{
  auto const raw_stats = read_raw_orc_statistics(src_info);

  parsed_orc_statistics result;
  result.column_names = raw_stats.column_names;

  auto parse_column_statistics = [](auto const& raw_col_stats) {
    orc::column_statistics stats_internal;
    orc::ProtobufReader(reinterpret_cast<const uint8_t*>(raw_col_stats.c_str()),
                        raw_col_stats.size())
      .read(stats_internal);
    return column_statistics(std::move(stats_internal));
  };

  std::transform(raw_stats.file_stats.cbegin(),
                 raw_stats.file_stats.cend(),
                 std::back_inserter(result.file_stats),
                 parse_column_statistics);

  for (auto const& raw_stripe_stats : raw_stats.stripes_stats) {
    result.stripes_stats.emplace_back();
    std::transform(raw_stripe_stats.cbegin(),
                   raw_stripe_stats.cend(),
                   std::back_inserter(result.stripes_stats.back()),
                   parse_column_statistics);
  }

  return result;
}

/**
 * @copydoc cudf::io::read_orc
 */
table_with_metadata read_orc(orc_reader_options const& options, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto reader = make_reader<detail_orc::reader>(options.get_source(), options, mr);

  return reader->read(options);
}

/**
 * @copydoc cudf::io::write_orc
 */
void write_orc(orc_writer_options const& options, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  namespace io_detail = cudf::io::detail;
  auto writer         = make_writer<detail_orc::writer>(
    options.get_sink(), options, io_detail::SingleWriteMode::YES, mr);

  writer->write(options.get_table());
}

/**
 * @copydoc cudf::io::orc_chunked_writer::orc_chunked_writer
 */
orc_chunked_writer::orc_chunked_writer(chunked_orc_writer_options const& op,
                                       rmm::mr::device_memory_resource* mr)
{
  namespace io_detail = cudf::io::detail;
  writer              = make_writer<detail_orc::writer>(
    op.get_sink(), op, io_detail::SingleWriteMode::NO, mr, rmm::cuda_stream_default);
}

/**
 * @copydoc cudf::io::orc_chunked_writer::write
 */
orc_chunked_writer& orc_chunked_writer::write(table_view const& table)
{
  CUDF_FUNC_RANGE();

  writer->write(table);

  return *this;
}

/**
 * @copydoc cudf::io::orc_chunked_writer::close
 */
void orc_chunked_writer::close()
{
  CUDF_FUNC_RANGE();

  writer->close();
}

using namespace cudf::io::detail::parquet;
namespace detail_parquet = cudf::io::detail::parquet;

table_with_metadata read_parquet(parquet_reader_options const& options,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto reader = make_reader<detail_parquet::reader>(options.get_source(), options, mr);

  return reader->read(options);
}

/**
 * @copydoc cudf::io::merge_rowgroup_metadata
 */
std::unique_ptr<std::vector<uint8_t>> merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>>& metadata_list)
{
  CUDF_FUNC_RANGE();
  return detail_parquet::writer::merge_rowgroup_metadata(metadata_list);
}

/**
 * @copydoc cudf::io::write_parquet
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet(parquet_writer_options const& options,
                                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  namespace io_detail = cudf::io::detail;

  auto writer = make_writer<detail_parquet::writer>(
    options.get_sink(), options, io_detail::SingleWriteMode::YES, mr, rmm::cuda_stream_default);

  writer->write(options.get_table());
  return writer->close(options.get_column_chunks_file_path());
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::parquet_chunked_writer
 */
parquet_chunked_writer::parquet_chunked_writer(chunked_parquet_writer_options const& op,
                                               rmm::mr::device_memory_resource* mr)
{
  namespace io_detail = cudf::io::detail;
  writer              = make_writer<detail_parquet::writer>(
    op.get_sink(), op, io_detail::SingleWriteMode::NO, mr, rmm::cuda_stream_default);
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::write
 */
parquet_chunked_writer& parquet_chunked_writer::write(table_view const& table)
{
  CUDF_FUNC_RANGE();

  writer->write(table);

  return *this;
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::close
 */
std::unique_ptr<std::vector<uint8_t>> parquet_chunked_writer::close(
  std::string const& column_chunks_file_path)
{
  CUDF_FUNC_RANGE();
  return writer->close(column_chunks_file_path);
}

}  // namespace io
}  // namespace cudf
