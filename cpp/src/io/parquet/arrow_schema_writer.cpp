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

class FieldPosition;

/**
 * @brief Function to construct a tree of arrow schema fields
 */
FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
                                     FieldPosition field_position,
                                     cudf::detail::LinkedColPtr const& col,
                                     column_in_metadata const& col_meta,
                                     single_write_mode const write_mode,
                                     bool const utc_timestamps);

// TODO: Copied over from ``writer_impl.cu``. Need to placed at a common location to avoid
// duplication.
inline bool is_col_nullable(cudf::detail::LinkedColPtr const& col,
                            column_in_metadata const& col_meta,
                            single_write_mode write_mode)
{
  if (col_meta.is_nullability_defined()) {
    CUDF_EXPECTS(col_meta.nullable() or col->null_count() == 0,
                 "Mismatch in metadata prescribed nullability and input column. "
                 "Metadata for input column with nulls cannot prescribe nullability = false");
    return col_meta.nullable();
  }
  // For chunked write, when not provided nullability, we assume the worst case scenario
  // that all columns are nullable.
  return write_mode == single_write_mode::NO or col->nullable();
}

// Helper class copied over from Arrow source. Do we need it even?
class FieldPosition {
 public:
  FieldPosition() : _parent(nullptr), _index(-1), _depth(0) {}

  FieldPosition child(int index) const { return {this, index}; }

  std::vector<int> path() const
  {
    std::vector<int> path(_depth);
    const FieldPosition* cur = this;
    for (int i = _depth - 1; i >= 0; --i) {
      path[i] = cur->_index;
      cur     = cur->_parent;
    }
    return path;
  }

 protected:
  FieldPosition(const FieldPosition* parent, int index)
    : _parent(parent), _index(index), _depth(parent->_depth + 1)
  {
  }

  const FieldPosition* _parent;
  int _index;
  int _depth;
};

/**
 * @brief Functor to convert cudf column metadata to arrow schema
 */
struct dispatch_to_flatbuf {
  FlatBufferBuilder& fbb;
  cudf::detail::LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  single_write_mode const write_mode;
  bool const utc_timestamps;
  FieldPosition& field_position;
  Offset& field_offset;
  flatbuf::Type& type_type;
  std::vector<FieldOffset>& children;

  template <typename T>
  std::enable_if_t<std::is_same_v<T, bool>, void> operator()()
  {
    type_type    = flatbuf::Type_Bool;
    field_offset = flatbuf::CreateBool(fbb).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int8_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 8, true).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int16_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 16, true).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int32_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 32, true).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, int64_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 64, true).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint8_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 8, false).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint16_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 16, false).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint32_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 32, false).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, uint64_t>, void> operator()()
  {
    type_type    = flatbuf::Type_Int;
    field_offset = flatbuf::CreateInt(fbb, 64, false).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, float>, void> operator()()
  {
    type_type    = flatbuf::Type_FloatingPoint;
    field_offset = flatbuf::CreateFloatingPoint(fbb, flatbuf::Precision::Precision_SINGLE).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, double>, void> operator()()
  {
    type_type    = flatbuf::Type_FloatingPoint;
    field_offset = flatbuf::CreateFloatingPoint(fbb, flatbuf::Precision::Precision_DOUBLE).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::string_view>, void> operator()()
  {
    type_type    = flatbuf::Type_Utf8View;
    field_offset = flatbuf::CreateUtf8View(fbb).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_D> or std::is_same_v<T, cudf::timestamp_s>,
                   void>
  operator()()
  {
    type_type    = flatbuf::Type_Timestamp;
    field_offset = flatbuf::CreateTimestamp(
                     fbb, flatbuf::TimeUnit_SECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
                     .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ms>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MILLISECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_us>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MICROSECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ns>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_NANOSECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_D> or std::is_same_v<T, cudf::duration_s>, void>
  operator()()
  {
    type_type    = flatbuf::Type_Duration;
    field_offset = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_SECOND).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ms>, void> operator()()
  {
    type_type    = flatbuf::Type_Duration;
    field_offset = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_MILLISECOND).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_us>, void> operator()()
  {
    type_type    = flatbuf::Type_Duration;
    field_offset = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_MICROSECOND).Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::duration_ns>, void> operator()()
  {
    type_type    = flatbuf::Type_Duration;
    field_offset = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_NANOSECOND).Union();
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()()
  {
    if (std::is_same_v<T, numeric::decimal128>) {
      type_type = flatbuf::Type_Decimal;
      field_offset =
        flatbuf::CreateDecimal(fbb, col_meta.get_decimal_precision(), col->type().scale(), 128)
          .Union();
    } else {
      CUDF_FAIL("fixed point type other than decimal128 not supported for arrow schema");
    }
  }

  template <typename T>
  std::enable_if_t<cudf::is_nested<T>(), void> operator()()
  {
    // TODO: Handle list and struct types. Remember, Lists are different in arrow schema and PQ
    // schema pq schema. List<int> in PQ schema:  "column_name" : { "list" : { "element" }} in
    // List<int> in arrow schema: "column_name" : { "list<element>" }
    // TODO: Arrow expects only 1 child for Lists and Structs. How and Why?
    std::transform(thrust::make_counting_iterator(0ul),
                   thrust::make_counting_iterator(col->children.size()),
                   std::back_inserter(children),
                   [&](auto const idx) {
                     return make_arrow_schema_fields(fbb,
                                                     field_position.child(idx),
                                                     col->children[idx],
                                                     col_meta.child(idx),
                                                     write_mode,
                                                     utc_timestamps);
                   });

    if (std::is_same_v<T, cudf::list_view>) {
      type_type    = flatbuf::Type_List;
      field_offset = flatbuf::CreateList(fbb).Union();
    } else if (std::is_same_v<T, cudf::struct_view>) {
      type_type    = flatbuf::Type_Struct_;
      field_offset = flatbuf::CreateStruct_(fbb).Union();
    } else {
      CUDF_FAIL("Unexpected nested type");
    }
  }

  template <typename T>
  std::enable_if_t<cudf::is_dictionary<T>(), void> operator()()
  {
    CUDF_FAIL("Dictionary columns are not supported for writing");
  }
};

FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
                                     FieldPosition field_position,
                                     cudf::detail::LinkedColPtr const& col,
                                     column_in_metadata const& col_meta,
                                     single_write_mode const write_mode,
                                     bool const utc_timestamps)
{
  Offset field_offset     = 0;
  flatbuf::Type type_type = flatbuf::Type_NONE;
  std::vector<FieldOffset> children;

  cudf::type_dispatcher(col->type(),
                        dispatch_to_flatbuf{fbb,
                                            col,
                                            col_meta,
                                            write_mode,
                                            utc_timestamps,
                                            field_position,
                                            field_offset,
                                            type_type,
                                            children});

  auto const fb_name          = fbb.CreateString(col_meta.get_name());
  auto const fb_children      = fbb.CreateVector(children.data(), children.size());
  auto const is_nullable      = is_col_nullable(col, col_meta, write_mode);
  DictionaryOffset dictionary = 0;

  // push to field offsets vector
  return flatbuf::CreateField(
    fbb, fb_name, is_nullable, type_type, field_offset, dictionary, fb_children);
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
std::string construct_arrow_schema_ipc_message(cudf::detail::LinkedColVector const& linked_columns,
                                               table_input_metadata const& metadata,
                                               single_write_mode const write_mode,
                                               bool const utc_timestamps)
{
  // Lambda function to convert int32 to a string of uint8 bytes
  auto const convert_int32_to_byte_string = [&](int32_t const value) {
    std::array<uint8_t, sizeof(int32_t)> buffer;
    std::memcpy(buffer.data(), &value, sizeof(int32_t));
    return std::string(reinterpret_cast<char*>(buffer.data()), buffer.size());
  };

  // intantiate a flatbuffer builder
  FlatBufferBuilder fbb;

  FieldPosition field_position;
  std::vector<FieldOffset> field_offsets;

  // populate field offsets (aka schema fields)
  std::transform(thrust::make_counting_iterator(0ul),
                 thrust::make_counting_iterator(linked_columns.size()),
                 std::back_inserter(field_offsets),
                 [&](auto const idx) {
                   return make_arrow_schema_fields(fbb,
                                                   field_position.child(idx),
                                                   linked_columns[idx],
                                                   metadata.column_metadata[idx],
                                                   write_mode,
                                                   utc_timestamps);
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
