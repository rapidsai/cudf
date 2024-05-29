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

/**
 * @brief Function to construct a tree of arrow schema fields
 */
FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
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

/**
 * @brief Functor to convert cudf column metadata to arrow schema
 */
struct dispatch_to_flatbuf {
  FlatBufferBuilder& fbb;
  cudf::detail::LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  single_write_mode const write_mode;
  bool const utc_timestamps;
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
    type_type = flatbuf::Type_Timestamp;
    // TODO: Verify if this is the correct logic
    field_offset = flatbuf::CreateTimestamp(
                     fbb, flatbuf::TimeUnit_SECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
                     .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ms>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    // TODO: Verify if this is the correct logic for UTC
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MILLISECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_us>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    // TODO: Verify if this is the correct logic for UTC
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MICROSECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  std::enable_if_t<std::is_same_v<T, cudf::timestamp_ns>, void> operator()()
  {
    type_type = flatbuf::Type_Timestamp;
    // TODO: Verify if this is the correct logic for UTC
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
    // TODO: cuDF-PQ writer supports d32 and d64 types not supported by Arrow without conversion.
    // See more: https://github.com/rapidsai/cudf/blob/branch-24.08/cpp/src/interop/to_arrow.cu#L155
    //
    if (std::is_same_v<T, numeric::decimal128>) {
      type_type = flatbuf::Type_Decimal;
      field_offset =
        flatbuf::CreateDecimal(fbb, col_meta.get_decimal_precision(), col->type().scale(), 128)
          .Union();
    } else {
      // TODO: Should we fail or just not write arrow:schema anymore?
      CUDF_FAIL("Fixed point types other than decimal128 are not supported for arrow schema");
    }
  }

  template <typename T>
  std::enable_if_t<cudf::is_nested<T>(), void> operator()()
  {
    // Lists are represented differently in arrow and cuDF.
    // cuDF representation: List<int>: "col_name" : { "list","element : int" } (2 children)
    // arrow schema representation: List<int>: "col_name" : { "list<item : int>" } (1 child)
    if constexpr (std::is_same_v<T, cudf::list_view>) {
      // Only need to process the second child (at idx = 1)
      children.emplace_back(make_arrow_schema_fields(
        fbb, col->children[1], col_meta.child(1), write_mode, utc_timestamps));
      type_type    = flatbuf::Type_List;
      field_offset = flatbuf::CreateList(fbb).Union();
    }
    // Traverse the struct in DFS manner and process children fields.
    else if constexpr (std::is_same_v<T, cudf::struct_view>) {
      std::transform(thrust::make_counting_iterator(0UL),
                     thrust::make_counting_iterator(col->children.size()),
                     std::back_inserter(children),
                     [&](auto const idx) {
                       return make_arrow_schema_fields(
                         fbb, col->children[idx], col_meta.child(idx), write_mode, utc_timestamps);
                     });
      type_type    = flatbuf::Type_Struct_;
      field_offset = flatbuf::CreateStruct_(fbb).Union();
    }
  }

  template <typename T>
  std::enable_if_t<cudf::is_dictionary<T>(), void> operator()()
  {
    // TODO: Implementing ``dictionary32`` would need ``DictionaryFieldMapper`` and
    // ``FieldPosition`` classes from arrow source to keep track of dictionary encoding paths.
    CUDF_FAIL("Dictionary columns are not supported for writing");
  }
};

FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
                                     cudf::detail::LinkedColPtr const& col,
                                     column_in_metadata const& col_meta,
                                     single_write_mode const write_mode,
                                     bool const utc_timestamps)
{
  Offset field_offset     = 0;
  flatbuf::Type type_type = flatbuf::Type_NONE;
  std::vector<FieldOffset> children;

  cudf::type_dispatcher(
    col->type(),
    dispatch_to_flatbuf{
      fbb, col, col_meta, write_mode, utc_timestamps, field_offset, type_type, children});

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

  // Instantiate a flatbuffer builder
  FlatBufferBuilder fbb;

  // Create an empty field offset vector
  std::vector<FieldOffset> field_offsets;

  // populate field offsets (aka schema fields)
  std::transform(
    thrust::make_counting_iterator(0ul),
    thrust::make_counting_iterator(linked_columns.size()),
    std::back_inserter(field_offsets),
    [&](auto const idx) {
      return make_arrow_schema_fields(
        fbb, linked_columns[idx], metadata.column_metadata[idx], write_mode, utc_timestamps);
    });

  // Build an arrow:schema flatbuffer using the field offset vector and use it as the header to
  // create an ipc message flatbuffer
  fbb.Finish(flatbuf::CreateMessage(
    fbb,
    flatbuf::MetadataVersion_V5,   /* Metadata version V5 (latest) */
    flatbuf::MessageHeader_Schema, /* Schema type message header */
    flatbuf::CreateSchema(
      fbb, flatbuf::Endianness::Endianness_Little, fbb.CreateVector(field_offsets))
      .Union(),                               /* Build an arrow:schema from the field vector */
    SCHEMA_HEADER_TYPE_IPC_MESSAGE_BODYLENGTH /* Body length is zero for schema type ipc message */
    ));

  // Construct the final string and store it here to use its view in base64_encode
  std::string const ipc_message =
    convert_int32_to_byte_string(IPC_CONTINUATION_TOKEN) +
    // Since the schema type ipc message doesn't have a body, the flatbuffer size is equal to the
    // ipc message's metadata length
    convert_int32_to_byte_string(fbb.GetSize()) +
    std::string(reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());

  // Encode the final ipc message string to base64 and return
  return cudf::io::detail::base64_encode(ipc_message);
}

}  // namespace cudf::io::parquet::detail
