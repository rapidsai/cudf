#include "cudf/io_readers.hpp"

#include "io/json/json_reader_impl.hpp"

namespace cudf {

JsonReader::JsonReader() noexcept = default;

JsonReader::JsonReader(JsonReader const &rhs) : impl_(std::make_unique<JsonReader::Impl>(rhs.impl_->getArgs())) {}

JsonReader &JsonReader::operator=(JsonReader const &rhs) {
  impl_.reset();
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

table JsonReader::read_byte_range(size_t byte_range_offset, size_t byte_range_size) {
  if (impl_) {
    return impl_->read_byte_range(byte_range_offset, byte_range_size);
  } else {
    return table();
  }
}

JsonReader::~JsonReader() = default;

} // namespace cudf