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
 * @file arrow_schema.hpp
 * @brief Arrow IPC schema writer implementation
 */

#pragma once

#include "io/parquet/parquet.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/utilities/base64_utilities.hpp"
#include "ipc/Message_generated.h"
#include "ipc/Schema_generated.h"

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

namespace flatbuf = cudf::io::parquet::flatbuf;

using FlatBufferBuilder = flatbuffers::FlatBufferBuilder;
using DictionaryOffset  = flatbuffers::Offset<flatbuf::DictionaryEncoding>;
using FieldOffset       = flatbuffers::Offset<flatbuf::Field>;
using Offset            = flatbuffers::Offset<void>;
using FBString          = flatbuffers::Offset<flatbuffers::String>;

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

std::vector<FieldOffset> make_field_offsets(host_span<SchemaElement const> parquet_schema)
{
  // MH: Get here
  std::vector<FieldOffset> field_offsets;
  FieldPosition pos;

  for (size_type i = 0; i < static_cast<size_type>(parquet_schema.size()); ++i) {
    FieldOffset offset;
    // FieldToFlatbufferVisitor field_visitor(fbb, mapper, pos.child(i));
    // field_visitor.GetResult(schema.field(i), &offset);
    field_offsets.push_back(offset);
  }
  return field_offsets;
}

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
  // Lambda function to convert int32 to a string of uint8 bytes
  auto const convert_int32_to_byte_string = [&](int32_t const value) {
    std::array<uint8_t, sizeof(int32_t)> buffer;
    std::memcpy(buffer.data(), &value, sizeof(int32_t));
    return std::string(reinterpret_cast<char*>(buffer.data()), buffer.size());
  };

  FlatBufferBuilder fbb;
  auto fb_offsets = fbb.CreateVector(make_field_offsets(parquet_schema));

  flatbuffers::Offset<flatbuf::Schema> const fb_schema =
    flatbuf::CreateSchema(fbb, flatbuf::Endianness::Endianness_Little, fb_offsets);

  auto const ipc_message_flatbuffer = flatbuf::CreateMessage(fbb,
                                                             flatbuf::MetadataVersion_V5,
                                                             flatbuf::MessageHeader_Schema,
                                                             fb_schema.Union(),
                                                             0 /* body_length */);
  fbb.Finish(ipc_message_flatbuffer);

  int32_t metadata_len = fbb.GetSize();

  // Store the final string here to pass its view to base64_encode
  std::string ipc_message =
    convert_int32_to_byte_string(IPC_CONTINUATION_TOKEN) +
    convert_int32_to_byte_string(metadata_len) +
    std::string(reinterpret_cast<char*>(fbb.GetBufferPointer()), metadata_len);

  // encode the final ipc message to base64 and return
  return cudf::io::detail::base64_encode(ipc_message);
}

}  // namespace cudf::io::parquet::detail
