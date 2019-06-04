#include "cudf/io_readers.hpp"

#include "io/json/json_reader_impl.hpp"

namespace cudf {

JsonReader::JsonReader(json_reader_args const &args){
  impl_ = std::make_unique<Impl>(args);
};

table JsonReader::read(){
  return impl_->read();
}

table JsonReader::read_byte_range(size_t byte_range_offset, size_t byte_range_size){
  return impl_->read_byte_range(byte_range_offset, byte_range_size);
}

JsonReader::~JsonReader() = default;

}