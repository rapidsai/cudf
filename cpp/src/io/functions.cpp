/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "io/orc/orc.hpp"

#include <cudf/detail/iterator.cuh>
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
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/resource_ref.hpp>

#include <algorithm>

namespace cudf::io {
// Returns builder for csv_reader_options
csv_reader_options_builder csv_reader_options::builder(source_info src)
{
  return csv_reader_options_builder{std::move(src)};
}

// Returns builder for csv_writer_options
csv_writer_options_builder csv_writer_options::builder(sink_info const& sink,
                                                       table_view const& table)
{
  return csv_writer_options_builder{sink, table};
}

// Returns builder for orc_reader_options
orc_reader_options_builder orc_reader_options::builder(source_info src)
{
  return orc_reader_options_builder{std::move(src)};
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
avro_reader_options_builder avro_reader_options::builder(source_info src)
{
  return avro_reader_options_builder(std::move(src));
}

// Returns builder for json_reader_options
json_reader_options_builder json_reader_options::builder(source_info src)
{
  return json_reader_options_builder(std::move(src));
}

// Returns builder for orc_writer_options
json_writer_options_builder json_writer_options::builder(sink_info const& sink,
                                                         table_view const& table)
{
  return json_writer_options_builder{sink, table};
}

// Returns builder for parquet_reader_options
parquet_reader_options_builder parquet_reader_options::builder(source_info src)
{
  return parquet_reader_options_builder{std::move(src)};
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

std::vector<std::unique_ptr<cudf::io::datasource>> make_datasources(source_info const& info,
                                                                    size_t range_offset = 0,
                                                                    size_t range_size   = 0)
{
  switch (info.type()) {
    case io_type::FILEPATH: {
      auto sources = std::vector<std::unique_ptr<cudf::io::datasource>>();
      for (auto const& filepath : info.filepaths()) {
        sources.emplace_back(cudf::io::datasource::create(filepath, range_offset, range_size));
      }
      return sources;
    }
    case io_type::HOST_BUFFER: return cudf::io::datasource::create(info.host_buffers());
    case io_type::DEVICE_BUFFER: return cudf::io::datasource::create(info.device_buffers());
    case io_type::USER_IMPLEMENTED: return cudf::io::datasource::create(info.user_sources());
    default: CUDF_FAIL("Unsupported source type");
  }
}

std::vector<std::unique_ptr<data_sink>> make_datasinks(sink_info const& info)
{
  switch (info.type()) {
    case io_type::FILEPATH: return cudf::io::data_sink::create(info.filepaths());
    case io_type::HOST_BUFFER: return cudf::io::data_sink::create(info.buffers());
    case io_type::VOID: {
      std::vector<std::unique_ptr<data_sink>> sinks;
      for (size_t i = 0; i < info.num_sinks(); ++i) {
        sinks.push_back(cudf::io::data_sink::create());
      }
      return sinks;
    }
    case io_type::USER_IMPLEMENTED: return cudf::io::data_sink::create(info.user_sinks());
    default: CUDF_FAIL("Unsupported sink type");
  }
}

}  // namespace

table_with_metadata read_avro(avro_reader_options const& options, rmm::device_async_resource_ref mr)
{
  namespace avro = cudf::io::detail::avro;

  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());

  CUDF_EXPECTS(datasources.size() == 1, "Only a single source is currently supported.");

  return avro::read_avro(std::move(datasources[0]), options, cudf::get_default_stream(), mr);
}

compression_type infer_compression_type(compression_type compression, source_info const& info)
{
  if (compression != compression_type::AUTO) { return compression; }

  if (info.type() != io_type::FILEPATH) { return compression_type::NONE; }

  auto filepath = info.filepaths()[0];

  // Attempt to infer from the file extension
  auto const pos = filepath.find_last_of('.');

  if (pos == std::string::npos) { return {}; }

  auto str_tolower = [](auto const& begin, auto const& end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  auto const ext = str_tolower(filepath.begin() + pos + 1, filepath.end());

  if (ext == "gz") { return compression_type::GZIP; }
  if (ext == "zip") { return compression_type::ZIP; }
  if (ext == "bz2") { return compression_type::BZIP2; }
  if (ext == "xz") { return compression_type::XZ; }

  return compression_type::NONE;
}

table_with_metadata read_json(json_reader_options options,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  return json::detail::read_json(datasources, options, stream, mr);
}

void write_json(json_writer_options const& options,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for JSON writing");

  return json::detail::write_json(  //
    sinks[0].get(),
    options.get_table(),
    options,
    stream,
    mr);
}

table_with_metadata read_csv(csv_reader_options options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  CUDF_EXPECTS(datasources.size() == 1, "Only a single source is currently supported.");

  return cudf::io::detail::csv::read_csv(  //
    std::move(datasources[0]),
    options,
    stream,
    mr);
}

// Freeform API wraps the detail writer class API
void write_csv(csv_writer_options const& options,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
{
  using namespace cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for CSV writing");

  return csv::write_csv(  //
    sinks[0].get(),
    options.get_table(),
    options.get_names(),
    options,
    stream,
    mr);
}

raw_orc_statistics read_raw_orc_statistics(source_info const& src_info,
                                           rmm::cuda_stream_view stream)
{
  // Get source to read statistics from
  std::unique_ptr<datasource> source;
  if (src_info.type() == io_type::FILEPATH) {
    CUDF_EXPECTS(src_info.filepaths().size() == 1, "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.filepaths()[0]);
  } else if (src_info.type() == io_type::HOST_BUFFER) {
    CUDF_EXPECTS(src_info.host_buffers().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.host_buffers()[0]);
  } else if (src_info.type() == io_type::DEVICE_BUFFER) {
    CUDF_EXPECTS(src_info.device_buffers().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.device_buffers()[0]);
  } else if (src_info.type() == io_type::USER_IMPLEMENTED) {
    CUDF_EXPECTS(src_info.user_sources().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.user_sources()[0]);
  } else {
    CUDF_FAIL("Unsupported source type");
  }

  orc::metadata metadata(source.get(), stream);

  // Initialize statistics to return
  raw_orc_statistics result;

  // Get column names
  for (auto i = 0; i < metadata.get_num_columns(); i++) {
    result.column_names.push_back(metadata.column_name(i));
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

column_statistics::column_statistics(orc::column_statistics&& cs)
{
  number_of_values = cs.number_of_values;
  has_null         = cs.has_null;
  if (cs.int_stats) {
    type_specific_stats = *cs.int_stats;
  } else if (cs.double_stats) {
    type_specific_stats = *cs.double_stats;
  } else if (cs.string_stats) {
    type_specific_stats = *cs.string_stats;
  } else if (cs.bucket_stats) {
    type_specific_stats = *cs.bucket_stats;
  } else if (cs.decimal_stats) {
    type_specific_stats = *cs.decimal_stats;
  } else if (cs.date_stats) {
    type_specific_stats = *cs.date_stats;
  } else if (cs.binary_stats) {
    type_specific_stats = *cs.binary_stats;
  } else if (cs.timestamp_stats) {
    type_specific_stats = *cs.timestamp_stats;
  }
}

parsed_orc_statistics read_parsed_orc_statistics(source_info const& src_info,
                                                 rmm::cuda_stream_view stream)
{
  auto const raw_stats = read_raw_orc_statistics(src_info, stream);

  parsed_orc_statistics result;
  result.column_names = raw_stats.column_names;

  auto parse_column_statistics = [](auto const& raw_col_stats) {
    orc::column_statistics stats_internal;
    orc::ProtobufReader(reinterpret_cast<uint8_t const*>(raw_col_stats.c_str()),
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
namespace {
orc_column_schema make_orc_column_schema(host_span<orc::SchemaType const> orc_schema,
                                         uint32_t column_id,
                                         std::string column_name)
{
  auto const& orc_col_schema = orc_schema[column_id];
  std::vector<orc_column_schema> children;
  children.reserve(orc_col_schema.subtypes.size());
  std::transform(
    orc_col_schema.subtypes.cbegin(),
    orc_col_schema.subtypes.cend(),
    cudf::detail::make_counting_transform_iterator(0,
                                                   [&names = orc_col_schema.fieldNames](size_t i) {
                                                     return i < names.size() ? names[i]
                                                                             : std::string{};
                                                   }),
    std::back_inserter(children),
    [&](auto& type, auto name) { return make_orc_column_schema(orc_schema, type, name); });

  return {std::move(column_name), orc_schema[column_id].kind, std::move(children)};
}
};  // namespace

orc_metadata read_orc_metadata(source_info const& src_info, rmm::cuda_stream_view stream)
{
  auto sources = make_datasources(src_info);

  CUDF_EXPECTS(sources.size() == 1, "Only a single source is currently supported.");
  auto const footer = orc::metadata(sources.front().get(), stream).ff;

  return {{make_orc_column_schema(footer.types, 0, "")},
          footer.numberOfRows,
          static_cast<size_type>(footer.stripes.size())};
}

/**
 * @copydoc cudf::io::read_orc
 */
table_with_metadata read_orc(orc_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());
  auto reader = std::make_unique<orc::detail::reader>(std::move(datasources), options, stream, mr);
  return reader->read();
}

/**
 * @copydoc cudf::io::write_orc
 */
void write_orc(orc_writer_options const& options, rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  CUDF_FUNC_RANGE();

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for ORC writing");

  auto writer = std::make_unique<orc::detail::writer>(
    std::move(sinks[0]), options, io_detail::single_write_mode::YES, stream);
  writer->write(options.get_table());
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       std::size_t pass_read_limit,
                                       size_type output_row_granularity,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : reader{std::make_unique<orc::detail::chunked_reader>(chunk_read_limit,
                                                         pass_read_limit,
                                                         output_row_granularity,
                                                         make_datasources(options.get_source()),
                                                         options,
                                                         stream,
                                                         mr)}
{
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       std::size_t pass_read_limit,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : reader{std::make_unique<orc::detail::chunked_reader>(chunk_read_limit,
                                                         pass_read_limit,
                                                         make_datasources(options.get_source()),
                                                         options,
                                                         stream,
                                                         mr)}
{
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : chunked_orc_reader(chunk_read_limit, 0UL, options, stream, mr)
{
}

// This destructor destroys the internal reader instance.
// Since the declaration of the internal `reader` object does not exist in the header, this
// destructor needs to be defined in a separate source file which can access to that object's
// declaration.
chunked_orc_reader::~chunked_orc_reader() = default;

bool chunked_orc_reader::has_next() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->has_next();
}

table_with_metadata chunked_orc_reader::read_chunk() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->read_chunk();
}

/**
 * @copydoc cudf::io::orc_chunked_writer::orc_chunked_writer
 */
orc_chunked_writer::orc_chunked_writer(chunked_orc_writer_options const& options,
                                       rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for ORC writing");

  writer = std::make_unique<orc::detail::writer>(
    std::move(sinks[0]), options, io_detail::single_write_mode::NO, stream);
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

using namespace cudf::io::parquet::detail;
namespace detail_parquet = cudf::io::parquet::detail;

table_with_metadata read_parquet(parquet_reader_options const& options,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());
  auto reader =
    std::make_unique<detail_parquet::reader>(std::move(datasources), options, stream, mr);

  return reader->read();
}

parquet_metadata read_parquet_metadata(source_info const& src_info)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(src_info);
  return detail_parquet::read_parquet_metadata(datasources);
}

/**
 * @copydoc cudf::io::merge_row_group_metadata
 */
std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
  std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list)
{
  CUDF_FUNC_RANGE();
  return detail_parquet::writer::merge_row_group_metadata(metadata_list);
}

table_input_metadata::table_input_metadata(table_view const& table)
{
  // Create a metadata hierarchy using `table`
  std::function<column_in_metadata(column_view const&)> get_children = [&](column_view const& col) {
    auto col_meta = column_in_metadata{};
    std::transform(
      col.child_begin(), col.child_end(), std::back_inserter(col_meta.children), get_children);
    return col_meta;
  };

  std::transform(
    table.begin(), table.end(), std::back_inserter(this->column_metadata), get_children);
}

table_input_metadata::table_input_metadata(table_metadata const& metadata)
{
  auto const& names = metadata.schema_info;

  // Create a metadata hierarchy with naming and nullability using `table_metadata`
  std::function<column_in_metadata(column_name_info const&)> process_node =
    [&](column_name_info const& name) {
      auto col_meta = column_in_metadata{name.name};
      if (name.is_nullable.has_value()) { col_meta.set_nullability(name.is_nullable.value()); }
      if (name.is_binary.value_or(false)) { col_meta.set_output_as_binary(true); }
      if (name.type_length.has_value()) { col_meta.set_type_length(name.type_length.value()); }
      std::transform(name.children.begin(),
                     name.children.end(),
                     std::back_inserter(col_meta.children),
                     process_node);
      return col_meta;
    };

  std::transform(
    names.begin(), names.end(), std::back_inserter(this->column_metadata), process_node);
}

/**
 * @copydoc cudf::io::write_parquet
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet(parquet_writer_options const& options,
                                                    rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  CUDF_FUNC_RANGE();

  auto sinks  = make_datasinks(options.get_sink());
  auto writer = std::make_unique<detail_parquet::writer>(
    std::move(sinks), options, io_detail::single_write_mode::YES, stream);

  writer->write(options.get_table(), options.get_partitions());

  return writer->close(options.get_column_chunks_file_paths());
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               parquet_reader_options const& options,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(
      chunk_read_limit, 0, make_datasources(options.get_source()), options, stream, mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               std::size_t pass_read_limit,
                                               parquet_reader_options const& options,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(chunk_read_limit,
                                                            pass_read_limit,
                                                            make_datasources(options.get_source()),
                                                            options,
                                                            stream,
                                                            mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::~chunked_parquet_reader
 */
chunked_parquet_reader::~chunked_parquet_reader() = default;

/**
 * @copydoc cudf::io::chunked_parquet_reader::has_next
 */
bool chunked_parquet_reader::has_next() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->has_next();
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::read_chunk
 */
table_with_metadata chunked_parquet_reader::read_chunk() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->read_chunk();
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::parquet_chunked_writer
 */
parquet_chunked_writer::parquet_chunked_writer(chunked_parquet_writer_options const& options,
                                               rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());

  writer = std::make_unique<detail_parquet::writer>(
    std::move(sinks), options, io_detail::single_write_mode::NO, stream);
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::write
 */
parquet_chunked_writer& parquet_chunked_writer::write(table_view const& table,
                                                      std::vector<partition_info> const& partitions)
{
  CUDF_FUNC_RANGE();

  writer->write(table, partitions);

  return *this;
}

/**
 * @copydoc cudf::io::parquet_chunked_writer::close
 */
std::unique_ptr<std::vector<uint8_t>> parquet_chunked_writer::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  CUDF_FUNC_RANGE();
  return writer->close(column_chunks_file_path);
}

void parquet_reader_options::set_row_groups(std::vector<std::vector<size_type>> row_groups)
{
  if ((!row_groups.empty()) and ((_skip_rows != 0) or _num_rows.has_value())) {
    CUDF_FAIL("row_groups can't be set along with skip_rows and num_rows");
  }

  _row_groups = std::move(row_groups);
}

void parquet_reader_options::set_skip_rows(int64_t val)
{
  CUDF_EXPECTS(val >= 0, "skip_rows cannot be negative");
  CUDF_EXPECTS(_row_groups.empty(), "skip_rows can't be set along with a non-empty row_groups");

  _skip_rows = val;
}

void parquet_reader_options::set_num_rows(size_type val)
{
  CUDF_EXPECTS(val >= 0, "num_rows cannot be negative");
  CUDF_EXPECTS(_row_groups.empty(), "num_rows can't be set along with a non-empty row_groups");

  _num_rows = val;
}

void parquet_writer_options::set_partitions(std::vector<partition_info> partitions)
{
  CUDF_EXPECTS(partitions.size() == _sink.num_sinks(),
               "Mismatch between number of sinks and number of partitions");
  _partitions = std::move(partitions);
}

void parquet_writer_options::set_key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  CUDF_EXPECTS(metadata.size() == _sink.num_sinks(),
               "Mismatch between number of sinks and number of metadata maps");
  _user_data = std::move(metadata);
}

void parquet_writer_options::set_column_chunks_file_paths(std::vector<std::string> file_paths)
{
  CUDF_EXPECTS(file_paths.size() == _sink.num_sinks(),
               "Mismatch between number of sinks and number of chunk paths to set");
  _column_chunks_file_paths = std::move(file_paths);
}

void parquet_writer_options::set_row_group_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(
    size_bytes >= 1024,
    "The maximum row group size cannot be smaller than the minimum page size, which is 1KB.");
  _row_group_size_bytes = size_bytes;
}

void parquet_writer_options::set_row_group_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum row group row count must be a positive integer.");
  _row_group_size_rows = size_rows;
}

void parquet_writer_options::set_max_page_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 1024, "The maximum page size cannot be smaller than 1KB.");
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum page size cannot exceed 2GB.");
  _max_page_size_bytes = size_bytes;
}

void parquet_writer_options::set_max_page_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum page row count must be a positive integer.");
  _max_page_size_rows = size_rows;
}

void parquet_writer_options::set_column_index_truncate_length(int32_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 0, "Column index truncate length cannot be negative.");
  _column_index_truncate_length = size_bytes;
}

void parquet_writer_options::set_dictionary_policy(dictionary_policy policy)
{
  _dictionary_policy = policy;
}

void parquet_writer_options::set_max_dictionary_size(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum dictionary size cannot exceed 2GB.");
  _max_dictionary_size = size_bytes;
}

void parquet_writer_options::set_max_page_fragment_size(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "Page fragment size must be a positive integer.");
  _max_page_fragment_size = size_rows;
}

parquet_writer_options_builder& parquet_writer_options_builder::partitions(
  std::vector<partition_info> partitions)
{
  options.set_partitions(std::move(partitions));
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  options.set_key_value_metadata(std::move(metadata));
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::column_chunks_file_paths(
  std::vector<std::string> file_paths)
{
  options.set_column_chunks_file_paths(std::move(file_paths));
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::dictionary_policy(
  enum dictionary_policy val)
{
  options.set_dictionary_policy(val);
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::max_dictionary_size(size_t val)
{
  options.set_max_dictionary_size(val);
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::max_page_fragment_size(
  size_type val)
{
  options.set_max_page_fragment_size(val);
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::write_v2_headers(bool enabled)
{
  options.enable_write_v2_headers(enabled);
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::sorting_columns(
  std::vector<sorting_column> sorting_columns)
{
  options._sorting_columns = std::move(sorting_columns);
  return *this;
}

void chunked_parquet_writer_options::set_key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  CUDF_EXPECTS(metadata.size() == _sink.num_sinks(),
               "Mismatch between number of sinks and number of metadata maps");
  _user_data = std::move(metadata);
}

void chunked_parquet_writer_options::set_row_group_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(
    size_bytes >= 1024,
    "The maximum row group size cannot be smaller than the minimum page size, which is 1KB.");
  _row_group_size_bytes = size_bytes;
}

void chunked_parquet_writer_options::set_row_group_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum row group row count must be a positive integer.");
  _row_group_size_rows = size_rows;
}

void chunked_parquet_writer_options::set_max_page_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 1024, "The maximum page size cannot be smaller than 1KB.");
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum page size cannot exceed 2GB.");
  _max_page_size_bytes = size_bytes;
}

void chunked_parquet_writer_options::set_max_page_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum page row count must be a positive integer.");
  _max_page_size_rows = size_rows;
}

void chunked_parquet_writer_options::set_column_index_truncate_length(int32_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 0, "Column index truncate length cannot be negative.");
  _column_index_truncate_length = size_bytes;
}

void chunked_parquet_writer_options::set_dictionary_policy(dictionary_policy policy)
{
  _dictionary_policy = policy;
}

void chunked_parquet_writer_options::set_max_dictionary_size(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum dictionary size cannot exceed 2GB.");
  _max_dictionary_size = size_bytes;
}

void chunked_parquet_writer_options::set_max_page_fragment_size(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "Page fragment size must be a positive integer.");
  _max_page_fragment_size = size_rows;
}

chunked_parquet_writer_options_builder& chunked_parquet_writer_options_builder::key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  options.set_key_value_metadata(std::move(metadata));
  return *this;
}

chunked_parquet_writer_options_builder& chunked_parquet_writer_options_builder::dictionary_policy(
  enum dictionary_policy val)
{
  options.set_dictionary_policy(val);
  return *this;
}

chunked_parquet_writer_options_builder& chunked_parquet_writer_options_builder::max_dictionary_size(
  size_t val)
{
  options.set_max_dictionary_size(val);
  return *this;
}

chunked_parquet_writer_options_builder& chunked_parquet_writer_options_builder::write_v2_headers(
  bool enabled)
{
  options.enable_write_v2_headers(enabled);
  return *this;
}

chunked_parquet_writer_options_builder& chunked_parquet_writer_options_builder::sorting_columns(
  std::vector<sorting_column> sorting_columns)
{
  options._sorting_columns = std::move(sorting_columns);
  return *this;
}

chunked_parquet_writer_options_builder&
chunked_parquet_writer_options_builder::max_page_fragment_size(size_type val)
{
  options.set_max_page_fragment_size(val);
  return *this;
}

}  // namespace cudf::io
