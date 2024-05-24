/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/**
 * @file arrow_schema.cpp
 * @brief Arrow IPC schema writer implementation
 */

#include "arrow_schema_writer.hpp"

namespace cudf::io::parquet::detail {

// Helper class copied over from Arrow source
class FieldPosition {
 public:
  FieldPosition() : parent_(nullptr), _index(-1), _depth(0) {}

  FieldPosition child(int index) const { return {this, index}; }

  std::vector<int> path() const
  {
    std::vector<int> path(_depth);
    const FieldPosition* cur = this;
    for (int i = _depth - 1; i >= 0; --i) {
      path[i] = cur->_index;
      cur     = cur->parent_;
    }
    return path;
  }

 protected:
  FieldPosition(const FieldPosition* parent, int index)
    : parent_(parent), _index(index), _depth(parent->_depth + 1)
  {
  }

  const FieldPosition* parent_;
  int _index;
  int _depth;
};

// Functor for cudf to flatbuf::type conversion
struct dispatch_to_flatbuf_type {};

/**
 * @brief Construct and return arrow schema from input parquet schema
 *
 * Recursively traverses through parquet schema to construct the arrow schema tree.
 * Serializes the arrow schema tree and stores it as the header (or metadata) of
 * an otherwise empty ipc message using flatbuffers. The ipc message is then prepended
 * with header size (padded for 16 byte alignment) and a continuation string. The final
 * string is base64 encoded and returned.
 */
std::string construct_arrow_schema_ipc_message(host_span<SchemaElement const> parquet_schema)
{
  // intantiate a flatbuffer builder
  FlatBufferBuilder fbb;

  // Lambda function to construct a tree of arrow schema fields
  std::function<FieldOffset(FieldPosition, int32_t const)> make_arrow_schema_fields =
    [&](FieldPosition pos, int32_t const schema_idx) -> FieldOffset {
    SchemaElement const schema_elem = parquet_schema[schema_idx];

    std::vector<FieldOffset> children{};

    std::transform(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(schema_elem.num_children),
                   std::back_inserter(children),
                   [&](auto const idx) {
                     return make_arrow_schema_fields(pos.child(idx), schema_elem.children_idx[idx]);
                   });

    auto type_type = flatbuf::Type_NONE;
    Offset type_offset;

    // TODO: Implement functor
    /*cudf::type_dispatcher(schema_elem.arrow_type.value_or(type_id::EMPTY),
                          dispatch_to_flatbuf_type{},
                          schema_elem,
                          type_offset,
                          type_type,
                          children);*/

    auto const fb_name     = fbb.CreateString(schema_elem.name);
    auto const fb_children = fbb.CreateVector(children.data(), children.size());
    auto const is_nullable = schema_elem.repetition_type == FieldRepetitionType::OPTIONAL or
                             schema_elem.repetition_type == FieldRepetitionType::REPEATED;
    DictionaryOffset dictionary = 0;

    // push to field offsets vector
    return flatbuf::CreateField(
      fbb, fb_name, is_nullable, type_type, type_offset, dictionary, fb_children);
  };

  // Lambda function to convert int32 to a string of uint8 bytes
  auto const convert_int32_to_byte_string = [&](int32_t const value) {
    std::array<uint8_t, sizeof(int32_t)> buffer;
    std::memcpy(buffer.data(), &value, sizeof(int32_t));
    return std::string(reinterpret_cast<char*>(buffer.data()), buffer.size());
  };

  // TODO: What to do with this?
  [[maybe_unused]] FieldPosition pos;
  std::vector<FieldOffset> field_offsets;

  // populate field offsets (aka schema fields)
  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(parquet_schema[0].num_children),
                 std::back_inserter(field_offsets),
                 [&](auto const idx) {
                   return make_arrow_schema_fields(pos.child(idx),
                                                   parquet_schema[0].children_idx[idx]);
                 });

  // Create a flatbuffer vector from the field offset vector
  auto const fb_offsets = fbb.CreateVector(field_offsets);

  // Create an arrow:schema flatbuffer
  flatbuffers::Offset<flatbuf::Schema> const fb_schema =
    flatbuf::CreateSchema(fbb, flatbuf::Endianness::Endianness_Little, fb_offsets);

  // Schema type message has zero length body
  constexpr int64_t bodylength = 0;

  // Create an ipc message flatbuffer
  auto const ipc_message_flatbuffer = flatbuf::CreateMessage(
    fbb, flatbuf::MetadataVersion_V5, flatbuf::MessageHeader_Schema, fb_schema.Union(), bodylength);

  // All done, finish building flatbuffers
  fbb.Finish(ipc_message_flatbuffer);

  // Since the ipc message doesn't have a body or other custom key value metadata,
  //  its size is equal to the size of its header (the schema flatbuffer)
  int32_t const metadata_len = fbb.GetSize();

  // Construct the final string and store in this variable here to use in base64_encode
  std::string const ipc_message =
    convert_int32_to_byte_string(IPC_CONTINUATION_TOKEN) +
    convert_int32_to_byte_string(metadata_len) +
    std::string(reinterpret_cast<char*>(fbb.GetBufferPointer()), metadata_len);

  // Encode the final ipc message string to base64 and return
  return cudf::io::detail::base64_encode(ipc_message);
}

}  // namespace cudf::io::parquet::detail
