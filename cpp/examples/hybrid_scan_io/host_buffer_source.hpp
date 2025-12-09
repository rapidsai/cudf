/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>

class host_buffer_source : public cudf::io::datasource {
 public:
  explicit host_buffer_source(cudf::host_span<std::byte const> h_buffer);

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  std::future<std::size_t> device_read_async(std::size_t offset,
                                             std::size_t size,
                                             uint8_t* dst,
                                             rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t size() const override;

 private:
  cudf::host_span<std::byte const> _h_buffer;
};
