/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <string>

namespace CUDF_EXPORT cudf {
namespace io::text {

/**
 * @brief Creates a data source capable of producing device-buffered views of a datasource.
 * @param data the datasource to be exposed as a data chunk source
 * @return the data chunk source for the provided datasource. It must not outlive the datasource
 *         used to construct it.
 */
std::unique_ptr<data_chunk_source> make_source(datasource& data);

/**
 * @brief Creates a data source capable of producing device-buffered views of the given string.
 * @param data the host data to be exposed as a data chunk source. Its lifetime must be at least as
 *             long as the lifetime of the returned data_chunk_source.
 * @return the data chunk source for the provided host data. It copies data from the host to the
 *         device.
 */
std::unique_ptr<data_chunk_source> make_source(host_span<char const> data);

/**
 * @brief Creates a data source capable of producing device-buffered views of the file
 * @param filename the filename of the file to be exposed as a data chunk source.
 * @return the data chunk source for the provided filename. It reads data from the file and copies
 *         it to the device.
 */
std::unique_ptr<data_chunk_source> make_source_from_file(std::string_view filename);

/**
 * @brief Creates a data source capable of producing device-buffered views of a BGZIP compressed
 *        file.
 * @param filename the filename of the BGZIP-compressed file to be exposed as a data chunk source.
 * @return the data chunk source for the provided filename. It reads data from the file and copies
 *         it to the device, where it will be decompressed.
 */
std::unique_ptr<data_chunk_source> make_source_from_bgzip_file(std::string_view filename);

/**
 * @brief Creates a data source capable of producing device-buffered views of a BGZIP compressed
 *        file with virtual record offsets.
 * @param filename the filename of the BGZIP-compressed file to be exposed as a data chunk source.
 * @param virtual_begin the virtual (Tabix) offset of the first byte to be read. Its upper 48 bits
 *                      describe the offset into the compressed file, its lower 16 bits describe the
 *                      block-local offset.
 * @param virtual_end the virtual (Tabix) offset one past the last byte to be read.
 * @return the data chunk source for the provided filename. It reads data from the file and copies
 *         it to the device, where it will be decompressed. The chunk source only returns data
 *         between the virtual offsets `virtual_begin` and `virtual_end`.
 */
std::unique_ptr<data_chunk_source> make_source_from_bgzip_file(std::string_view filename,
                                                               uint64_t virtual_begin,
                                                               uint64_t virtual_end);

/**
 * @brief Creates a data source capable of producing views of the given device string scalar
 * @param data the device data to be exposed as a data chunk source. Its lifetime must be at least
 *             as long as the lifetime of the returned data_chunk_source.
 * @return the data chunk source for the provided host data. It does not create any copies.
 */
std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data);

}  // namespace io::text
}  // namespace CUDF_EXPORT cudf
