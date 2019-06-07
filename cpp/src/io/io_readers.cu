/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "cudf/io_readers.hpp"

#include "io/json/json_reader_impl.hpp"
#include "io/csv/csv_reader_impl.hpp"

namespace cudf {

JsonReader::JsonReader() noexcept = default;

JsonReader::JsonReader(JsonReader const &rhs) : impl_(std::make_unique<JsonReader::Impl>(rhs.impl_->getArgs())) {}

JsonReader &JsonReader::operator=(JsonReader const &rhs) {
  impl_ = std::make_unique<JsonReader::Impl>(rhs.impl_->getArgs());
  return *this;
}

JsonReader::JsonReader(JsonReader &&rhs) : impl_(std::move(rhs.impl_)) {}

JsonReader &JsonReader::operator=(JsonReader &&rhs) {
  impl_ = std::move(rhs.impl_);
  return *this;
}

JsonReader::JsonReader(json_reader_args const &args) : impl_(std::make_unique<Impl>(args)) {}

table JsonReader::read() {
  if (impl_) {
    return impl_->read();
  } else {
    return table();
  }
}

table JsonReader::read_byte_range(size_t offset, size_t size) {
  if (impl_) {
    return impl_->read_byte_range(offset, size);
  } else {
    return table();
  }
}

JsonReader::~JsonReader() = default;


CsvReader::CsvReader() noexcept = default;

CsvReader::CsvReader(CsvReader const &rhs) : impl_(std::make_unique<CsvReader::Impl>(rhs.impl_->getArgs())) {}

CsvReader &CsvReader::operator=(CsvReader const &rhs) {
  impl_ = std::make_unique<CsvReader::Impl>(rhs.impl_->getArgs());
  return *this;
}

CsvReader::CsvReader(CsvReader &&rhs) : impl_(std::move(rhs.impl_)) {}

CsvReader &CsvReader::operator=(CsvReader &&rhs) {
  impl_ = std::move(rhs.impl_);
  return *this;
}

CsvReader::CsvReader(csv_reader_args const &args) : impl_(std::make_unique<Impl>(args)) {}

table CsvReader::read() {
  if (impl_) {
    return impl_->read();
  } else {
    return table();
  }
}

CsvReader::~CsvReader() = default;

} // namespace cudf
