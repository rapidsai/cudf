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

#include <utility>

namespace CUDF_EXPORT cudf {
namespace io {

/**
 * @addtogroup io_cutable
 * @{
 * @file
 * @brief CUTable binary format APIs for serialization and deserialization
 */

/**
 * @brief Simple binary file format header for CUTable
 *
 * The CUTable format stores a table in a simple binary layout:
 * - Magic number (4 bytes): "CUDF"
 * - Version (4 bytes): uint32_t format version (currently 1)
 * - Metadata length (8 bytes): uint64_t size of the metadata buffer in bytes
 * - Data length (8 bytes): uint64_t size of the data buffer in bytes
 * - Metadata (variable): serialized column metadata from pack()
 * - Data (variable): contiguous device data from pack()
 */
struct cutable_header {
  static constexpr uint32_t magic_number = 0x46445543;  ///< "CUDF" in little-endian
  static constexpr uint32_t version      = 1;           ///< Format version

  uint32_t magic;            ///< Magic number for format validation
  uint32_t format_version;   ///< Format version number
  uint64_t metadata_length;  ///< Length of metadata buffer in bytes
  uint64_t data_length;      ///< Length of data buffer in bytes
};

class cutable_writer_options_builder;

/**
 * @brief Settings for `write_cutable()`.
 */
class cutable_writer_options {
  sink_info _sink;
  table_view _table;

  friend cutable_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit cutable_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _table(std::move(table))
  {
  }

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit cutable_writer_options() = default;

  /**
   * @brief Create builder to create `cutable_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build cutable_writer_options
   */
  static cutable_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const { return _sink; }

  /**
   * @brief Returns table that would be written to output.
   *
   * @return Table that would be written to output
   */
  [[nodiscard]] table_view const& get_table() const { return _table; }

  /**
   * @brief Sets sink info.
   *
   * @param sink The sink info.
   */
  void set_sink(sink_info sink) { _sink = std::move(sink); }
};

/**
 * @brief Class to build `cutable_writer_options`.
 */
class cutable_writer_options_builder {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit cutable_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit cutable_writer_options_builder(sink_info const& sink, table_view const& table)
    : _options(sink, table)
  {
  }

  /**
   * @brief Build `cutable_writer_options`.
   *
   * @return The constructed `cutable_writer_options` object
   */
  [[nodiscard]] cutable_writer_options build() const { return _options; }

 private:
  cutable_writer_options _options;
};

class cutable_reader_options_builder;

/**
 * @brief Settings for `read_cutable()`.
 */
class cutable_reader_options {
  source_info _source;

  friend cutable_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read cutable file
   */
  explicit cutable_reader_options(source_info src) : _source{std::move(src)} {}

 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit cutable_reader_options() = default;

  /**
   * @brief Creates a `cutable_reader_options_builder` to build `cutable_reader_options`.
   *
   * @param src Source information to read cutable file
   * @return Builder to build reader options
   */
  static cutable_reader_options_builder builder(source_info src = source_info{});

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const { return _source; }

  /**
   * @brief Sets source info.
   *
   * @param src The source info.
   */
  void set_source(source_info src) { _source = std::move(src); }
};

/**
 * @brief Class to build `cutable_reader_options`.
 */
class cutable_reader_options_builder {
 public:
  /**
   * @brief Default constructor.
   *
   * This has been added since Cython requires a default constructor to create objects on stack.
   */
  explicit cutable_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param src source information used to read cutable file
   */
  explicit cutable_reader_options_builder(source_info src) : _options(std::move(src)) {}

  /**
   * @brief Build `cutable_reader_options`.
   *
   * @return The constructed `cutable_reader_options` object
   */
  [[nodiscard]] cutable_reader_options build() const { return _options; }

 private:
  cutable_reader_options _options;
};

/** @} */  // end of group
}  // namespace io

namespace io::experimental {

/**
 * @addtogroup io_cutable
 * @{
 */

/**
 * @brief Write a table using the CUTable binary format.
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
 * @param options Options specifying the sink and table to write
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all device allocations
 */
void write_cutable(cutable_writer_options const& options,
                   rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                   rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Read a table in CUTable binary format.
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
 * @param options Options specifying the source to read from
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr An optional memory resource to use for all device allocations
 * @return A packed_table containing the deserialized table view and its backing data
 */
packed_table read_cutable(
  cutable_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io::experimental
}  // namespace CUDF_EXPORT cudf
