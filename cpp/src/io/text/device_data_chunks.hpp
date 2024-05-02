/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator device_span<char const>() const override { return _data; }

 private:
  rmm::device_uvector<char> _data;
};

}  // namespace cudf::io::text
