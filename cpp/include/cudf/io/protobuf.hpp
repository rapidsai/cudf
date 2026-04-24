/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::protobuf {

/**
 * @brief Protobuf field encoding types.
 */
enum class proto_encoding : int {
  DEFAULT     = 0,  ///< Standard varint encoding
  FIXED       = 1,  ///< Fixed-width encoding (32-bit or 64-bit)
  ZIGZAG      = 2,  ///< ZigZag encoding for signed integers
  ENUM_STRING = 3,  ///< Enum field decoded as string
};

/**
 * @brief Get the integer value of a proto_encoding.
 */
CUDF_HOST_DEVICE constexpr int encoding_value(proto_encoding encoding)
{
  return static_cast<int>(encoding);
}

/// Maximum protobuf field number (29-bit).
constexpr int MAX_FIELD_NUMBER = (1 << 29) - 1;

/**
 * @brief Protobuf wire types.
 */
enum class proto_wire_type : int {
  VARINT = 0,  ///< Variable-length integer
  I64BIT = 1,  ///< 64-bit fixed
  LEN    = 2,  ///< Length-delimited
  SGROUP = 3,  ///< Start group (deprecated)
  EGROUP = 4,  ///< End group (deprecated)
  I32BIT = 5,  ///< 32-bit fixed
};

/**
 * @brief Get the integer value of a proto_wire_type.
 */
CUDF_HOST_DEVICE constexpr int wire_type_value(proto_wire_type wire_type)
{
  return static_cast<int>(wire_type);
}

/// Maximum supported nesting depth for nested protobuf messages.
constexpr int MAX_NESTING_DEPTH = 10;

/**
 * @brief Descriptor for a single field in a (possibly nested) protobuf schema.
 *
 * Fields are organized in a flat array where parent-child relationships are
 * expressed via @p parent_idx. Top-level fields have `parent_idx == -1`.
 */
struct nested_field_descriptor {
  int field_number;           ///< Protobuf field number
  int parent_idx;             ///< Index of parent field in schema (-1 for top-level)
  int depth;                  ///< Nesting depth (0 for top-level)
  proto_wire_type wire_type;  ///< Expected wire type
  cudf::type_id output_type;  ///< Output cudf type
  proto_encoding encoding;    ///< Encoding type
  bool is_repeated;           ///< Whether this field is repeated (array)
  bool is_required;           ///< Whether this field is required (proto2)
  bool has_default_value;     ///< Whether this field has a default value
};

/**
 * @brief Context for decoding protobuf messages.
 *
 * Contains the schema (as a flat array of field descriptors), default values,
 * and enum metadata needed for decoding.
 */
struct decode_protobuf_options {
  std::vector<nested_field_descriptor> schema;  ///< Flat array of field descriptors
  std::vector<int64_t> default_ints;            ///< Default integer values per field
  std::vector<double> default_floats;           ///< Default float values per field
  std::vector<bool> default_bools;              ///< Default boolean values per field
  std::vector<cudf::detail::host_vector<uint8_t>>
    default_strings;  ///< Default string values per field
  std::vector<cudf::detail::host_vector<int32_t>>
    enum_valid_values;  ///< Valid enum numbers per field
  std::vector<std::vector<cudf::detail::host_vector<uint8_t>>>
    enum_names;         ///< UTF-8 enum names per field
  bool fail_on_errors;  ///< If true, throw on malformed messages; otherwise return nulls
};

/**
 * @brief View into a single field's metadata from a decode_protobuf_options.
 */
struct protobuf_field_meta_view {
  nested_field_descriptor const& schema;
  cudf::data_type const output_type;
  int64_t default_int;
  double default_float;
  bool default_bool;
  cudf::detail::host_vector<uint8_t> const& default_string;
  cudf::detail::host_vector<int32_t> const& enum_valid_values;
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_names;
};

/**
 * @brief Decode serialized protobuf messages into a struct column.
 *
 * Takes a LIST<UINT8> column where each row contains a serialized protobuf message,
 * and decodes it into a STRUCT column according to the provided schema.
 *
 * Supports nested messages (up to 10 levels), repeated fields (as LIST columns),
 * enum-as-string conversion, default values, and required field checking.
 *
 * @param binary_input LIST<INT8> or LIST<UINT8> column of serialized protobuf messages
 * @param options Decoding options including schema, defaults, and enum metadata
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A STRUCT column containing the decoded protobuf fields
 *
 * @throws cudf::logic_error if the schema is invalid
 * @throws cudf::logic_error if fail_on_errors is true and a message cannot be decoded
 */
std::unique_ptr<cudf::column> decode_protobuf(
  cudf::column_view const& binary_input,
  decode_protobuf_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace io::protobuf
}  // namespace CUDF_EXPORT cudf
