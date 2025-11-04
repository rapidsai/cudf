/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace io {

/**
 * @addtogroup io_table
 * @{
 * @file
 * @brief Table binary format APIs for serialization and deserialization
 */

/**
 * @brief Simple binary file format header
 *
 * The table format stores a table in a simple binary layout:
 * - Magic number (4 bytes): "CUDF"
 * - Version (4 bytes): uint32_t format version (currently 1)
 * - Metadata length (8 bytes): uint64_t size of the metadata buffer in bytes
 * - Data length (8 bytes): uint64_t size of the data buffer in bytes
 * - Metadata (variable): serialized column metadata from pack()
 * - Data (variable): contiguous device data from pack()
 */
struct table_format_header {
  static constexpr uint32_t magic_number = 0x46445543;  ///< "CUDF" in little-endian
  static constexpr uint32_t version      = 1;           ///< Format version

  uint32_t magic;            ///< Magic number for format validation
  uint32_t format_version;   ///< Format version number
  uint64_t metadata_length;  ///< Length of metadata buffer in bytes
  uint64_t data_length;      ///< Length of data buffer in bytes
};

/**
 * @brief Write a table using the table binary format.
 *
 * This function uses `cudf::pack` to serialize a table into a contiguous format,
 * then writes it to the specified sink with a simple header containing metadata
 * and data lengths.
 *
 * The output format consists of:
 * 1. A fixed-size header (24 bytes) containing:
 *    - Magic number for format validation
 *    - Version number for compatibility
 *    - Metadata buffer size
 *    - Data buffer size
 * 2. The metadata buffer (host-side)
 * 3. The data buffer (device-side, copied to sink)
 *
 * @throws cudf::logic_error If the sink cannot be written to
 *
 * @param input The table_view to write
 * @param sink_info The sink_info specifying output location
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all device allocations
 */
void write_table(cudf::table_view const& input,
                 sink_info const& sink_info,
                 rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Read a table in table binary format.
 *
 * This function reads the header from the datasource, validates the format,
 * and uses `cudf::unpack` to deserialize the table.
 *
 * Returns a `packed_table` containing a `table_view` and the underlying `packed_columns`
 * data. This is a zero-copy operation - the table_view points directly into the
 * contiguous memory buffers owned by the packed_columns.
 *
 * It is the caller's responsibility to ensure the table_view does not outlive
 * the packed_columns data.
 *
 * @throws cudf::logic_error If the header is invalid or corrupted
 * @throws cudf::logic_error If the format version is not supported
 *
 * @param source_info The source_info specifying input location
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all device allocations
 * @return A packed_table containing the deserialized table view and its backing data
 */
packed_table read_table(
  source_info const& source_info,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
