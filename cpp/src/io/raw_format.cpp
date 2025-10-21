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

#include "cudf/detail/nvtx/ranges.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/raw_format.hpp>
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
 * @brief Validate the raw format header
 *
 * @param header The header to validate
 * @throws cudf::logic_error If the header is invalid
 */
void validate_header(raw_format_header const& header)
{
  CUDF_EXPECTS(header.magic == raw_format_header::magic_number,
               "Invalid magic number in raw format header");
  CUDF_EXPECTS(header.format_version == raw_format_header::version,
               "Unsupported raw format version");
}

}  // anonymous namespace

void write_raw(cudf::table_view const& input,
               sink_info const& sink_info,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Create data_sink from sink_info based on type
  std::unique_ptr<data_sink> sink;
  switch (sink_info.type()) {
    case io_type::FILEPATH: sink = data_sink::create(sink_info.filepaths()[0]); break;
    case io_type::HOST_BUFFER: sink = data_sink::create(sink_info.buffers()[0]); break;
    case io_type::VOID: sink = data_sink::create(); break;
    case io_type::USER_IMPLEMENTED: sink = data_sink::create(sink_info.user_sinks()[0]); break;
    default: CUDF_FAIL("Unsupported sink type for raw format");
  }

  // Pack the table into contiguous memory
  auto packed = cudf::pack(input, stream, mr);

  // Create and populate the header
  raw_format_header header;
  header.magic           = raw_format_header::magic_number;
  header.format_version  = raw_format_header::version;
  header.metadata_length = packed.metadata->size();
  header.data_length     = packed.gpu_data->size();

  // Write the header
  sink->host_write(&header, sizeof(raw_format_header));

  // Write the metadata
  sink->host_write(packed.metadata->data(), header.metadata_length);

  // Write the data
  // Always copy data to host and then write (avoids kvikIO)
  auto host_buffer = cudf::detail::make_host_vector<uint8_t>(header.data_length, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_buffer.data(),
                                packed.gpu_data->data(),
                                header.data_length,
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  sink->host_write(host_buffer.data(), header.data_length);

  sink->flush();
}

packed_table read_raw(source_info const& source_info,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(source_info.num_sources() == 1, "Raw format only supports single source");

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
    default: CUDF_FAIL("Unsupported source type for raw format");
  }

  // Read the header
  raw_format_header header;
  auto header_size = sizeof(raw_format_header);
  CUDF_EXPECTS(source->size() >= header_size,
               "File too small to contain a valid raw format header");

  source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));

  // Validate the header
  validate_header(header);

  // Calculate offsets
  size_t metadata_offset = header_size;
  size_t data_offset     = metadata_offset + header.metadata_length;

  // Validate file size
  CUDF_EXPECTS(source->size() >= data_offset + header.data_length,
               "File too small for the specified metadata and data sizes");

  // Read metadata into host memory
  auto metadata = std::make_unique<std::vector<uint8_t>>(header.metadata_length);
  source->host_read(metadata_offset, header.metadata_length, metadata->data());

  // Allocate device memory for the data
  auto gpu_data = std::make_unique<rmm::device_buffer>(header.data_length, stream, mr);

  // Read data into device memory
  if (source->supports_device_read() && source->is_device_read_preferred(header.data_length)) {
    source->device_read(
      data_offset, header.data_length, static_cast<uint8_t*>(gpu_data->data()), stream);
  } else {
    // Read to host first, then copy to device
    auto host_buffer = cudf::detail::make_host_vector<uint8_t>(header.data_length, stream);
    source->host_read(data_offset, header.data_length, host_buffer.data());
    CUDF_CUDA_TRY(cudaMemcpyAsync(gpu_data->data(),
                                  host_buffer.data(),
                                  header.data_length,
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
  }

  // Synchronize to ensure data is ready
  stream.synchronize();

  // Create packed_columns structure
  packed_columns packed(std::move(metadata), std::move(gpu_data));

  // Unpack into a table_view
  auto unpacked_view = cudf::unpack(packed);

  // Return packed_table with the view and the data
  // The table_view points into the packed_columns memory (zero-copy)
  return packed_table{unpacked_view, std::move(packed)};
}

}  // namespace io
}  // namespace cudf
