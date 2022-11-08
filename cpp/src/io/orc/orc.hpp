/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/io/datasource.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/utilities/error.hpp>
#include <io/comp/io_uncomp.hpp>

#include <thrust/optional.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace orc {

static constexpr uint32_t block_header_size = 3;

struct PostScript {
  uint64_t footerLength       = 0;     // the length of the footer section in bytes
  CompressionKind compression = NONE;  // the kind of generic compression used
  uint32_t compressionBlockSize{};     // the maximum size of each compression chunk
  std::vector<uint32_t> version;       // the version of the writer [major, minor]
  uint64_t metadataLength = 0;         // the length of the metadata section in bytes
  std::string magic       = "";        // the fixed string "ORC"
};

struct StripeInformation {
  uint64_t offset       = 0;  // the start of the stripe within the file
  uint64_t indexLength  = 0;  // the length of the indexes in bytes
  uint64_t dataLength   = 0;  // the length of the data in bytes
  uint32_t footerLength = 0;  // the length of the footer in bytes
  uint32_t numberOfRows = 0;  // the number of rows in the stripe
};

struct SchemaType {
  TypeKind kind = INVALID_TYPE_KIND;  // the kind of this type
  std::vector<uint32_t> subtypes;  // the type ids of any subcolumns for list, map, struct, or union
  std::vector<std::string> fieldNames;  // the list of field names for struct
  std::optional<uint32_t>
    maximumLength;  // the maximum length of the type for varchar or char in UTF-8 characters
  std::optional<uint32_t> precision;  // the precision for decimal
  std::optional<uint32_t> scale;      // the scale for decimal
};

struct UserMetadataItem {
  std::string name;   // the user defined key
  std::string value;  // the user defined binary value as string
};

using ColStatsBlob = std::vector<uint8_t>;  // Column statistics blob

struct FileFooter {
  uint64_t headerLength  = 0;              // the length of the file header in bytes (always 3)
  uint64_t contentLength = 0;              // the length of the file header and body in bytes
  std::vector<StripeInformation> stripes;  // the information about the stripes
  std::vector<SchemaType> types;           // the schema information
  std::vector<UserMetadataItem> metadata;  // the user metadata that was added
  uint64_t numberOfRows = 0;               // the total number of rows in the file
  std::vector<ColStatsBlob> statistics;    // Column statistics blobs
  uint32_t rowIndexStride = 0;             // the maximum number of rows in each index entry
};

struct Stream {
  StreamKind kind = INVALID_STREAM_KIND;
  std::optional<uint32_t> column_id;  // ORC column id (different from column index in the table!)
  uint64_t length = 0;                // the number of bytes in the stream

  // Returns index of the column in the table, if any
  // Stream of the 'column 0' does not have a corresponding column in the table
  [[nodiscard]] std::optional<uint32_t> column_index() const noexcept
  {
    return column_id.value_or(0) > 0 ? std::optional<uint32_t>{*column_id - 1}
                                     : std::optional<uint32_t>{};
  }
};

struct ColumnEncoding {
  ColumnEncodingKind kind = INVALID_ENCODING_KIND;
  uint32_t dictionarySize = 0;  // for dictionary encodings, record the size of the dictionary
};

struct StripeFooter {
  std::vector<Stream> streams;          // the location of each stream
  std::vector<ColumnEncoding> columns;  // the encoding of each column
  std::string writerTimezone = "";      // time zone of the writer
};

/**
 * @brief Contains per-column ORC statistics.
 *
 * At most one of the `***_statistics` members has a value.
 */
struct column_statistics {
  std::optional<uint64_t> number_of_values;
  std::optional<integer_statistics> int_stats;
  std::optional<double_statistics> double_stats;
  std::optional<string_statistics> string_stats;
  std::optional<bucket_statistics> bucket_stats;
  std::optional<decimal_statistics> decimal_stats;
  std::optional<date_statistics> date_stats;
  std::optional<binary_statistics> binary_stats;
  std::optional<timestamp_statistics> timestamp_stats;
  std::optional<bool> has_null;
};

struct StripeStatistics {
  std::vector<ColStatsBlob> colStats;  // Column statistics blobs
};

struct Metadata {
  std::vector<StripeStatistics> stripeStats;
};

int inline constexpr encode_field_number(int field_number, ProtofType field_type) noexcept
{
  return (field_number * 8) + static_cast<int>(field_type);
}

namespace {
template <typename base_t,
          std::enable_if_t<!std::is_arithmetic_v<base_t> and !std::is_enum_v<base_t>>* = nullptr>
int static constexpr encode_field_number_base(int field_number) noexcept
{
  return encode_field_number(field_number, ProtofType::FIXEDLEN);
}

template <typename base_t,
          std::enable_if_t<std::is_integral_v<base_t> or std::is_enum_v<base_t>>* = nullptr>
int static constexpr encode_field_number_base(int field_number) noexcept
{
  return encode_field_number(field_number, ProtofType::VARINT);
}

template <typename base_t, std::enable_if_t<std::is_same_v<base_t, float>>* = nullptr>
int static constexpr encode_field_number_base(int field_number) noexcept
{
  return encode_field_number(field_number, ProtofType::FIXED32);
}

template <typename base_t, std::enable_if_t<std::is_same_v<base_t, double>>* = nullptr>
int static constexpr encode_field_number_base(int field_number) noexcept
{
  return encode_field_number(field_number, ProtofType::FIXED64);
}
};  // namespace

template <typename T,
          std::enable_if_t<!std::is_class_v<T> or std::is_same_v<T, std::string>>* = nullptr>
int constexpr encode_field_number(int field_number) noexcept
{
  return encode_field_number_base<T>(field_number);
}

// containers change the field number encoding
template <typename T,
          std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type>>>* = nullptr>
int constexpr encode_field_number(int field_number) noexcept
{
  return encode_field_number_base<T>(field_number);
}

// optional fields don't change the field number encoding
template <typename T,
          std::enable_if_t<std::is_same_v<T, std::optional<typename T::value_type>>>* = nullptr>
int constexpr encode_field_number(int field_number) noexcept
{
  return encode_field_number_base<typename T::value_type>(field_number);
}

/**
 * @brief Class for parsing Orc's Protocol Buffers encoded metadata
 */
class ProtobufReader {
 public:
  ProtobufReader(const uint8_t* base, size_t len) : m_base(base), m_cur(base), m_end(base + len) {}

  template <typename T>
  void read(T& s)
  {
    read(s, m_end - m_cur);
  }
  void read(PostScript&, size_t maxlen);
  void read(FileFooter&, size_t maxlen);
  void read(StripeInformation&, size_t maxlen);
  void read(SchemaType&, size_t maxlen);
  void read(UserMetadataItem&, size_t maxlen);
  void read(StripeFooter&, size_t maxlen);
  void read(Stream&, size_t maxlen);
  void read(ColumnEncoding&, size_t maxlen);
  void read(integer_statistics&, size_t maxlen);
  void read(double_statistics&, size_t maxlen);
  void read(string_statistics&, size_t maxlen);
  void read(bucket_statistics&, size_t maxlen);
  void read(decimal_statistics&, size_t maxlen);
  void read(date_statistics&, size_t maxlen);
  void read(binary_statistics&, size_t maxlen);
  void read(timestamp_statistics&, size_t maxlen);
  void read(column_statistics&, size_t maxlen);
  void read(StripeStatistics&, size_t maxlen);
  void read(Metadata&, size_t maxlen);

 private:
  template <int index>
  friend class FunctionSwitchImpl;

  void skip_bytes(size_t bytecnt)
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }

  template <typename T>
  T get();

  void skip_struct_field(int t);

  template <typename T, typename... Operator>
  void function_builder(T& s, size_t maxlen, std::tuple<Operator...>& op);

  uint32_t read_field_size(const uint8_t* end);

  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    value = get<T>();
  }

  template <typename T, std::enable_if_t<std::is_enum_v<T>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    value = static_cast<T>(get<uint32_t>());
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, std::string>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    auto const size = read_field_size(end);
    value.assign(reinterpret_cast<const char*>(m_cur), size);
    m_cur += size;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, std::vector<std::string>>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    auto const size = read_field_size(end);
    value.emplace_back(reinterpret_cast<const char*>(m_cur), size);
    m_cur += size;
  }

  template <typename T,
            std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type>> and
                             !std::is_same_v<std::string, typename T::value_type>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    auto const size = read_field_size(end);
    value.emplace_back();
    read(value.back(), size);
  }

  template <typename T,
            std::enable_if_t<std::is_same_v<T, std::optional<typename T::value_type>>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    typename T::value_type contained_value;
    read_field(contained_value, end);
    value = std::optional<typename T::value_type>{std::move(contained_value)};
  }

  template <typename T>
  auto read_field(T& value, const uint8_t* end) -> decltype(read(value, 0))
  {
    auto const size = read_field_size(end);
    read(value, size);
  }

  template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  void read_field(T& value, const uint8_t* end)
  {
    memcpy(&value, m_cur, sizeof(T));
    m_cur += sizeof(T);
  }

  template <typename T>
  void read_packed_field(T& value, const uint8_t* end)
  {
    auto const len       = get<uint32_t>();
    auto const field_end = std::min(m_cur + len, end);
    while (m_cur < field_end)
      value.push_back(get<typename T::value_type>());
  }

  template <typename T>
  void read_raw_field(T& value, const uint8_t* end)
  {
    auto const size = read_field_size(end);
    value.emplace_back(m_cur, m_cur + size);
    m_cur += size;
  }

  template <typename T>
  struct field_reader {
    int const encoded_field_number;
    T& output_value;

    field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(ProtobufReader* pbr, const uint8_t* end)
    {
      pbr->read_field(output_value, end);
    }
  };

  template <typename T>
  struct packed_field_reader {
    int const encoded_field_number;
    T& output_value;

    packed_field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(ProtobufReader* pbr, const uint8_t* end)
    {
      pbr->read_packed_field(output_value, end);
    }
  };

  template <typename T>
  struct raw_field_reader {
    int const encoded_field_number;
    T& output_value;

    raw_field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(ProtobufReader* pbr, const uint8_t* end)
    {
      pbr->read_raw_field(output_value, end);
    }
  };

  const uint8_t* const m_base;
  const uint8_t* m_cur;
  const uint8_t* const m_end;

 public:
  /**
   * @brief Returns a field reader object of correct type, based on the `field_value`
   * type.
   *
   * @tparam Type of the field (inferred from `field_value` type)
   * @param field_number The field number of the field to be read
   * @param field_value Reference to the object the field reader will write to
   * @return the field reader object of the right type
   */
  template <typename T>
  static auto make_field_reader(int field_number, T& field_value)
  {
    return field_reader<T>(field_number, field_value);
  }

  /**
   * @brief Returns a reader object for packed fields, based on the `field_value` type.
   *
   * @tparam Type of the field (inferred from `field_value` type)
   * @param field_number The field number of the field to be read
   * @param field_value Reference to the object the field reader will write to
   * @return the packed field reader object of the right type
   */
  template <typename T>
  static auto make_packed_field_reader(int field_number, T& field_value)
  {
    return packed_field_reader<T>(field_number, field_value);
  }

  /**
   * @brief Returns a field reader that does not decode data, with type based on the `field_value`
   * type.
   *
   * @tparam Type of the field (inferred from `field_value` type)
   * @param field_number The field number of the field to be read
   * @param field_value Reference to the object the field reader will write to
   * @return the raw field reader object of the right type
   */
  template <typename T>
  static auto make_raw_field_reader(int field_number, T& field_value)
  {
    return raw_field_reader<T>(field_number, field_value);
  }
};

template <>
inline uint8_t ProtobufReader::get<uint8_t>()
{
  return (m_cur < m_end) ? *m_cur++ : 0;
};

template <>
inline bool ProtobufReader::get<bool>()
{
  return static_cast<bool>(get<uint8_t>());
};

template <>
inline uint32_t ProtobufReader::get<uint32_t>()
{
  uint32_t v = 0;
  for (uint32_t l = 0;; l += 7) {
    uint32_t c = get<uint8_t>();
    v |= (c & 0x7f) << l;
    if (c < 0x80) return v;
  }
}

template <>
inline uint64_t ProtobufReader::get<uint64_t>()
{
  uint64_t v = 0;
  for (uint64_t l = 0;; l += 7) {
    uint64_t c = get<uint8_t>();
    v |= (c & 0x7f) << l;
    if (c < 0x80) return v;
  }
}

template <typename T>
auto decode_zigzag(T u)
{
  using signed_t = std::make_signed_t<T>;
  return static_cast<signed_t>((u >> 1u) ^ -static_cast<signed_t>(u & 1));
}

template <>
inline int32_t ProtobufReader::get<int32_t>()
{
  return decode_zigzag(get<uint32_t>());
}

template <>
inline int64_t ProtobufReader::get<int64_t>()
{
  return decode_zigzag(get<uint64_t>());
}

/**
 * @brief Class for encoding Orc's metadata with Protocol Buffers
 */
class ProtobufWriter {
 public:
  ProtobufWriter() { m_buf = nullptr; }
  ProtobufWriter(std::vector<uint8_t>* output) { m_buf = output; }
  uint32_t put_byte(uint8_t v)
  {
    m_buf->push_back(v);
    return 1;
  }
  template <typename T>
  uint32_t put_bytes(host_span<T const> values)
  {
    static_assert(sizeof(T) == 1);
    m_buf->reserve(m_buf->size() + values.size());
    m_buf->insert(m_buf->end(), values.begin(), values.end());
    return values.size();
  }
  uint32_t put_uint(uint64_t v)
  {
    int l = 1;
    while (v > 0x7f) {
      put_byte(static_cast<uint8_t>(v | 0x80));
      v >>= 7;
      l++;
    }
    put_byte(static_cast<uint8_t>(v));
    return l;
  }

  uint32_t varint_size(uint64_t val)
  {
    auto len = 1u;
    while (val > 0x7f) {
      val >>= 7;
      ++len;
    }
    return len;
  }

  uint32_t put_int(int64_t v)
  {
    int64_t s = (v < 0);
    return put_uint(((v ^ -s) << 1) + s);
  }
  void put_row_index_entry(int32_t present_blk,
                           int32_t present_ofs,
                           int32_t data_blk,
                           int32_t data_ofs,
                           int32_t data2_blk,
                           int32_t data2_ofs,
                           TypeKind kind,
                           ColStatsBlob const* stats);

 public:
  size_t write(const PostScript&);
  size_t write(const FileFooter&);
  size_t write(const StripeInformation&);
  size_t write(const SchemaType&);
  size_t write(const UserMetadataItem&);
  size_t write(const StripeFooter&);
  size_t write(const Stream&);
  size_t write(const ColumnEncoding&);
  size_t write(const StripeStatistics&);
  size_t write(const Metadata&);

 protected:
  std::vector<uint8_t>* m_buf;
  struct ProtobufFieldWriter;
};

/**
 * @brief Class for decompressing Orc data blocks using the CPU
 */

class OrcDecompressor {
 public:
  OrcDecompressor(CompressionKind kind, uint32_t blockSize);

  /**
   * @brief ORC block decompression
   *
   * @param src compressed data
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return decompressed data
   */
  host_span<uint8_t const> decompress_blocks(host_span<uint8_t const> src,
                                             rmm::cuda_stream_view stream);
  [[nodiscard]] uint32_t GetLog2MaxCompressionRatio() const { return m_log2MaxRatio; }
  [[nodiscard]] uint32_t GetMaxUncompressedBlockSize(uint32_t block_len) const
  {
    return std::min(block_len << m_log2MaxRatio, m_blockSize);
  }
  [[nodiscard]] compression_type compression() const { return _compression; }
  [[nodiscard]] uint32_t GetBlockSize() const { return m_blockSize; }

 protected:
  compression_type _compression;
  uint32_t m_log2MaxRatio = 24;  // log2 of maximum compression ratio
  uint32_t m_blockSize;
  std::vector<uint8_t> m_buf;
};

/**
 * @brief Stores orc id for each column and number of children in that column.
 *
 * @code{.pseudo}
 * Consider following data where a struct has two members and a list column
 * {"struct": [{"a": 1, "b": 2}, {"a":3, "b":5}], "list":[[1, 2], [2, 3]]}
 *
 * `orc_column_meta` for struct column would be
 * id = 0
 * num_children = 2
 *
 * `orc_column_meta` for list column would be
 * id = 3
 * num_children = 1
 * @endcode
 *
 */
struct orc_column_meta {
  size_type id;            // orc id for the column
  size_type num_children;  // number of children at the same level of nesting in case of struct
};

/**
 * @brief Stores column's validity map and null count
 */
struct column_validity_info {
  uint32_t* valid_map_base;
  uint32_t null_count;
};

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 */
class metadata {
  using OrcStripeInfo = std::pair<const StripeInformation*, const StripeFooter*>;

 public:
  struct stripe_source_mapping {
    int source_idx;
    std::vector<OrcStripeInfo> stripe_info;
  };

 public:
  explicit metadata(datasource* const src, rmm::cuda_stream_view stream);

  [[nodiscard]] size_t get_total_rows() const { return ff.numberOfRows; }
  [[nodiscard]] int get_num_stripes() const { return ff.stripes.size(); }
  [[nodiscard]] int get_num_columns() const { return ff.types.size(); }
  /**
   * @brief Returns the name of the column with the given ID.
   *
   * Name might not be unique in the ORC file, since columns with different parents are allowed to
   * have the same names.
   */
  [[nodiscard]] std::string const& column_name(size_type column_id) const
  {
    CUDF_EXPECTS(column_id < get_num_columns(), "Out of range column id provided");
    return column_names[column_id];
  }
  /**
   * @brief Returns the full name of the column with the given ID - includes the ancestor columns
   * names.
   *
   * Each column in the ORC file has a unique path.
   */
  [[nodiscard]] std::string const& column_path(size_type column_id) const
  {
    CUDF_EXPECTS(column_id < get_num_columns(), "Out of range column id provided");
    return column_paths[column_id];
  }
  [[nodiscard]] int get_row_index_stride() const { return ff.rowIndexStride; }

  /**
   * @brief Returns the ID of the parent column of the given column.
   */
  [[nodiscard]] size_type parent_id(size_type column_id) const
  {
    return parents.at(column_id).value().id;
  }

  /**
   * @brief Returns the index the given column has in its parent's children list.
   */
  [[nodiscard]] size_type field_index(size_type column_id) const
  {
    return parents.at(column_id).value().field_idx;
  }

  /**
   * @brief Returns whether the given column has a parent.
   */
  [[nodiscard]] size_type column_has_parent(size_type column_id) const
  {
    return parents.at(column_id).has_value();
  }

 public:
  PostScript ps;
  FileFooter ff;
  Metadata md;
  std::vector<StripeFooter> stripefooters;
  std::unique_ptr<OrcDecompressor> decompressor;
  datasource* const source;

 private:
  struct column_parent {
    // parent's ID
    size_type id;
    // Index of this column in the parent's list of children
    size_type field_idx;
    column_parent(size_type parent_id, size_type field_idx) : id{parent_id}, field_idx{field_idx} {}
  };
  void init_parent_descriptors();
  std::vector<std::optional<column_parent>> parents;

  void init_column_names();
  std::vector<std::string> column_names;
  std::vector<std::string> column_paths;
};

/**
 * @brief `column_device_view` and additional, ORC specific, information on the column.
 */
struct orc_column_device_view : public column_device_view {
  __device__ orc_column_device_view(column_device_view col, thrust::optional<uint32_t> parent_idx)
    : column_device_view{col}, parent_index{parent_idx}
  {
  }
  thrust::optional<uint32_t> parent_index;
  bitmask_type const* pushdown_mask = nullptr;
};

/**
 * @brief Range of rows within a single rowgroup.
 */
struct rowgroup_rows {
  size_type begin;
  size_type end;
  [[nodiscard]] constexpr auto size() const noexcept { return end - begin; }
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
