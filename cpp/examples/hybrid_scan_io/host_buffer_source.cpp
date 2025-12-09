/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "host_buffer_source.hpp"

#include <cudf/io/datasource.hpp>

host_buffer_source::host_buffer_source(cudf::host_span<std::byte const> h_buffer)
  : _h_buffer{h_buffer}
{
}

size_t host_buffer_source::host_read(size_t offset, size_t size, uint8_t* dst)
{
  auto const count = std::min(size, this->size() - offset);
  std::memcpy(dst, _h_buffer.data() + offset, count);
  return size;
}

std::unique_ptr<cudf::io::datasource::buffer> host_buffer_source::host_read(size_t offset,
                                                                            size_t size)
{
  auto const count = std::min(size, this->size() - offset);
  return std::make_unique<non_owning_buffer>(
    reinterpret_cast<uint8_t const*>(_h_buffer.data() + offset), count);
}

[[nodiscard]] bool host_buffer_source::supports_device_read() const { return true; }

std::future<std::size_t> host_buffer_source::device_read_async(std::size_t offset,
                                                               std::size_t size,
                                                               uint8_t* dst,
                                                               rmm::cuda_stream_view stream)
{
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    dst, _h_buffer.data() + offset, size, cudaMemcpyKind::cudaMemcpyDefault, stream));
  std::promise<std::size_t> p;
  auto future = p.get_future();
  p.set_value(size);
  return future;
}

[[nodiscard]] std::size_t host_buffer_source::size() const
{
  return _h_buffer.size();
}

;
