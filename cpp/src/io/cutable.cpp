/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf/detail/nvtx/ranges.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/cutable.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstring>
#include <vector>

namespace cudf {
namespace io {

namespace {

/**
 * @brief Validate the cutable header
 *
 * @param header The header to validate
 * @throws cudf::logic_error If the header is invalid
 */
void validate_header(cutable_header const& header)
{
  CUDF_EXPECTS(header.magic == cutable_header::magic_number,
               "Invalid magic number in cutable header");
  CUDF_EXPECTS(header.format_version == cutable_header::version,
               "Unsupported cutable format version");
}

}  // anonymous namespace

cutable_writer_options_builder cutable_writer_options::builder(sink_info const& sink,
                                                               table_view const& table)
{
  return cutable_writer_options_builder(sink, table);
}

cutable_reader_options_builder cutable_reader_options::builder(source_info src)
{
  return cutable_reader_options_builder(std::move(src));
}

}  // namespace io

namespace io::experimental {

void write_cutable(cutable_writer_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const& sink_info = options.get_sink();
  auto const& input     = options.get_table();

  // Create data_sink from sink_info based on type
  std::unique_ptr<data_sink> sink;
  switch (sink_info.type()) {
    case io_type::FILEPATH: sink = data_sink::create(sink_info.filepaths()[0]); break;
    case io_type::HOST_BUFFER: sink = data_sink::create(sink_info.buffers()[0]); break;
    case io_type::VOID: sink = data_sink::create(); break;
    case io_type::USER_IMPLEMENTED: sink = data_sink::create(sink_info.user_sinks()[0]); break;
    default: CUDF_FAIL("Unsupported sink type for cutable format");
  }

  // Pack the table into contiguous memory
  auto const packed = cudf::pack(input, stream, mr);

  // Create and populate the header
  cutable_header header;
  header.magic           = cutable_header::magic_number;
  header.format_version  = cutable_header::version;
  header.metadata_length = packed.metadata->size();
  header.data_length     = packed.gpu_data->size();

  // Write the header
  sink->host_write(&header, sizeof(cutable_header));

  // Write the metadata
  sink->host_write(packed.metadata->data(), header.metadata_length);

  // Write the data
  if (sink->is_device_write_preferred(header.data_length)) {
    sink->device_write(packed.gpu_data->data(), header.data_length, stream);
  } else {
    auto host_buffer = cudf::detail::make_host_vector(
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(packed.gpu_data->data()),
                                       header.data_length},
      stream);
    sink->host_write(host_buffer.data(), header.data_length);
  }

  sink->flush();
}

packed_table read_cutable(cutable_reader_options const& options,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const& source_info = options.get_source();
  CUDF_EXPECTS(source_info.num_sources() == 1, "CUTable format only supports single source");

  // Create datasource from source_info based on type
  std::unique_ptr<datasource> source;
  switch (source_info.type()) {
    case io_type::FILEPATH: source = datasource::create(source_info.filepaths()[0]); break;
    case io_type::HOST_BUFFER: source = datasource::create(source_info.host_buffers()[0]); break;
    case io_type::DEVICE_BUFFER:
      source = datasource::create(source_info.device_buffers()[0]);
      break;
    case io_type::USER_IMPLEMENTED:
      source = datasource::create(source_info.user_sources()[0]);
      break;
    default: CUDF_FAIL("Unsupported source type for cutable format");
  }

  // Read the header
  cutable_header header;
  auto header_size = sizeof(cutable_header);
  CUDF_EXPECTS(source->size() >= header_size, "File too small to contain a valid cutable header");

  source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));

  // Validate the header
  validate_header(header);

  // Calculate offsets
  size_t metadata_offset = header_size;
  size_t data_offset     = metadata_offset + header.metadata_length;

  CUDF_EXPECTS(source->size() >= data_offset + header.data_length,
               "File too small for the specified metadata and data sizes");

  // Create packed_columns and read directly into it
  packed_columns packed;
  packed.metadata->resize(header.metadata_length);
  source->host_read(metadata_offset, header.metadata_length, packed.metadata->data());

  packed.gpu_data = std::make_unique<rmm::device_buffer>(header.data_length, stream, mr);
  if (source->is_device_read_preferred(header.data_length)) {
    source->device_read(
      data_offset, header.data_length, static_cast<uint8_t*>(packed.gpu_data->data()), stream);
  } else {
    auto host_buffer = source->host_read(data_offset, header.data_length);
    CUDF_CUDA_TRY(cudaMemcpyAsync(packed.gpu_data->data(),
                                  host_buffer->data(),
                                  header.data_length,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  }

  auto unpacked_view = cudf::unpack(packed);

  return packed_table{unpacked_view, std::move(packed)};
}

}  // namespace io::experimental
}  // namespace cudf
