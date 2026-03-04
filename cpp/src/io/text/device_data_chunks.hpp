/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/text/data_chunk_source.hpp>

namespace cudf::io::text {

class device_span_data_chunk : public device_data_chunk {
 public:
  device_span_data_chunk(device_span<char const> data) : _data(data) {}

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator device_span<char const>() const override { return _data; }

 private:
  device_span<char const> _data;
};

class device_uvector_data_chunk : public device_data_chunk {
 public:
  device_uvector_data_chunk(rmm::device_uvector<char>&& data) : _data(std::move(data)) {}

  device_uvector_data_chunk(device_uvector_data_chunk const&)            = delete;
  device_uvector_data_chunk& operator=(device_uvector_data_chunk const&) = delete;
  device_uvector_data_chunk(device_uvector_data_chunk&&)                 = default;
  device_uvector_data_chunk& operator=(device_uvector_data_chunk&&)      = default;

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator device_span<char const>() const override { return _data; }

 private:
  rmm::device_uvector<char> _data;
};

}  // namespace cudf::io::text
