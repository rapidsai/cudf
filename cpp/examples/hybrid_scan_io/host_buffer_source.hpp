/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>

/**
 * @brief In-memory host datasource that allows copy to device.
 *
 * @note This class is different from libcudf's internal `cudf::io::host_buffer_source` which does
 * not allow copy to device.
 */
class host_buffer_source : public cudf::io::datasource {
 public:
  /**
   * @brief Constructs a host buffer data source from a host buffer that contains the Parquet file
   * data
   *
   * @param h_buffer A host buffer containing the Parquet file data
   */
  explicit host_buffer_source(cudf::host_span<std::byte const> h_buffer);

  /**
   * @brief This override is required by the base class `cudf::io::datasource`
   *
   * @param offset Offset at which to read the in-memory host data
   * @param size Number of bytes to read
   * @return A copy of the host buffer
   */
  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override;

  /**
   * @brief This override is required by the base class `cudf::io::datasource`
   *
   * @param offset Offset at which to read the in-memory host data
   * @param size Number of bytes to read
   * @param dst Preallocated host buffer to read into
   * @return Number of bytes that have been read
   */
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  /**
   * @brief This override is required by the base class `cudf::io::datasource`
   *
   * @return Whether this data source supports reading into device buffer
   */
  [[nodiscard]] bool supports_device_read() const override;

  /**
   * @brief Copies data to a preallocated device buffer
   *
   * @param offset Offset at which to read the in-memory host data
   * @param size Number of bytes to read
   * @param dst Preallocated device buffer to read into
   * @param stream CUDA stream
   * @return A future containing the size of H2D copy. The future has a ready state.
   */
  std::future<std::size_t> device_read_async(std::size_t offset,
                                             std::size_t size,
                                             uint8_t* dst,
                                             rmm::cuda_stream_view stream) override;

  /**
   * @brief This override is required by the base class `cudf::io::datasource`
   *
   * @return The size of the in-memory host buffer
   */
  [[nodiscard]] std::size_t size() const override;

 private:
  cudf::host_span<std::byte const> _h_buffer;
};
