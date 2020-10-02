/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @file writer_impl.cu
 * @brief cuDF-IO ORC writer class implementation
 */

#include "writer_impl.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstring>
#include <utility>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace orc {
using namespace cudf::io::orc;
using namespace cudf::io;

namespace {
/**
 * @brief Helper for pinned host memory
 **/
template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

/**
 * @brief Function that translates GDF compression to ORC compression
 **/
orc::CompressionKind to_orc_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return orc::CompressionKind::SNAPPY;
    case compression_type::NONE: return orc::CompressionKind::NONE;
    default: CUDF_EXPECTS(false, "Unsupported compression type"); return orc::CompressionKind::NONE;
  }
}

/**
 * @brief Function that translates GDF dtype to ORC datatype
 **/
constexpr orc::TypeKind to_orc_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8: return TypeKind::BYTE;
    case cudf::type_id::INT16: return TypeKind::SHORT;
    case cudf::type_id::INT32: return TypeKind::INT;
    case cudf::type_id::INT64: return TypeKind::LONG;
    case cudf::type_id::FLOAT32: return TypeKind::FLOAT;
    case cudf::type_id::FLOAT64: return TypeKind::DOUBLE;
    case cudf::type_id::BOOL8: return TypeKind::BOOLEAN;
    case cudf::type_id::TIMESTAMP_DAYS: return TypeKind::DATE;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return TypeKind::TIMESTAMP;
    case cudf::type_id::STRING: return TypeKind::STRING;
    default: return TypeKind::INVALID_TYPE_KIND;
  }
}

/**
 * @brief Function that translates time unit to nanoscale multiple
 **/
template <typename T>
constexpr T to_clockscale(cudf::type_id timestamp_id)
{
  switch (timestamp_id) {
    case cudf::type_id::TIMESTAMP_SECONDS: return 9;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 6;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 3;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    default: return 0;
  }
}

}  // namespace

/**
 * @brief Helper kernel for converting string data/offsets into nvstrdesc
 * REMOVEME: Once we eliminate the legacy readers/writers, the kernels could be
 * made to use the native offset+data layout.
 **/
__global__ void stringdata_to_nvstrdesc(gpu::nvstrdesc_s *dst,
                                        const size_type *offsets,
                                        const char *strdata,
                                        const uint32_t *nulls,
                                        size_type column_size)
{
  size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < column_size) {
    uint32_t is_valid = (nulls) ? (nulls[row >> 5] >> (row & 0x1f)) & 1 : 1;
    size_t count;
    const char *ptr;
    if (is_valid) {
      size_type cur  = offsets[row];
      size_type next = offsets[row + 1];
      ptr            = strdata + cur;
      count          = (next > cur) ? next - cur : 0;
    } else {
      ptr   = nullptr;
      count = 0;
    }
    dst[row].ptr   = ptr;
    dst[row].count = count;
  }
}

/**
 * @brief Helper class that adds ORC-specific column info
 **/
class orc_column_view {
 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   **/
  explicit orc_column_view(size_t id,
                           size_t str_id,
                           column_view const &col,
                           const table_metadata *metadata,
                           cudaStream_t stream)
    : _id(id),
      _str_id(str_id),
      _string_type(col.type().id() == type_id::STRING),
      _type_width(_string_type ? 0 : cudf::size_of(col.type())),
      _data_count(col.size()),
      _null_count(col.null_count()),
      _data(col.head<uint8_t>() + col.offset() * _type_width),
      _nulls(col.nullable() ? col.null_mask() : nullptr),
      _clockscale(to_clockscale<uint8_t>(col.type().id())),
      _type_kind(to_orc_type(col.type().id()))
  {
    if (_string_type && _data_count > 0) {
      strings_column_view view{col};
      _indexes = rmm::device_buffer(_data_count * sizeof(gpu::nvstrdesc_s), stream);
      stringdata_to_nvstrdesc<<<((_data_count - 1) >> 8) + 1, 256, 0, stream>>>(
        reinterpret_cast<gpu::nvstrdesc_s *>(_indexes.data()),
        view.offsets().data<size_type>() + view.offset(),
        view.chars().data<char>(),
        _nulls,
        _data_count);
      _data = _indexes.data();
      CUDA_TRY(cudaStreamSynchronize(stream));
    }
    // Generating default name if name isn't present in metadata
    if (metadata && _id < metadata->column_names.size()) {
      _name = metadata->column_names[_id];
    } else {
      _name = "_col" + std::to_string(_id);
    }
  }

  auto is_string() const noexcept { return _string_type; }
  void set_dict_stride(size_t stride) noexcept { dict_stride = stride; }
  auto get_dict_stride() const noexcept { return dict_stride; }

  /**
   * @brief Function that associates an existing dictionary chunk allocation
   **/
  void attach_dict_chunk(gpu::DictionaryChunk *host_dict, gpu::DictionaryChunk *dev_dict)
  {
    dict   = host_dict;
    d_dict = dev_dict;
  }
  auto host_dict_chunk(size_t rowgroup)
  {
    assert(_string_type);
    return &dict[rowgroup * dict_stride + _str_id];
  }
  auto device_dict_chunk() const { return d_dict; }

  /**
   * @brief Function that associates an existing stripe dictionary allocation
   **/
  void attach_stripe_dict(gpu::StripeDictionary *host_stripe_dict,
                          gpu::StripeDictionary *dev_stripe_dict)
  {
    stripe_dict   = host_stripe_dict;
    d_stripe_dict = dev_stripe_dict;
  }
  auto host_stripe_dict(size_t stripe) const
  {
    assert(_string_type);
    return &stripe_dict[stripe * dict_stride + _str_id];
  }
  auto device_stripe_dict() const { return d_stripe_dict; }

  size_t type_width() const noexcept { return _type_width; }
  size_t data_count() const noexcept { return _data_count; }
  size_t null_count() const noexcept { return _null_count; }
  bool nullable() const noexcept { return (_nulls != nullptr); }
  void const *data() const noexcept { return _data; }
  uint32_t const *nulls() const noexcept { return _nulls; }
  uint8_t clockscale() const noexcept { return _clockscale; }

  void set_orc_encoding(ColumnEncodingKind e) { _encoding_kind = e; }
  auto orc_kind() const noexcept { return _type_kind; }
  auto orc_encoding() const noexcept { return _encoding_kind; }
  auto orc_name() const noexcept { return _name; }

 private:
  // Identifier within set of columns and string columns, respectively
  size_t _id        = 0;
  size_t _str_id    = 0;
  bool _string_type = false;

  size_t _type_width     = 0;
  size_t _data_count     = 0;
  size_t _null_count     = 0;
  void const *_data      = nullptr;
  uint32_t const *_nulls = nullptr;
  uint8_t _clockscale    = 0;

  // ORC-related members
  std::string _name{};
  TypeKind _type_kind;
  ColumnEncodingKind _encoding_kind;

  // String dictionary-related members
  rmm::device_buffer _indexes;
  size_t dict_stride                       = 0;
  gpu::DictionaryChunk const *dict         = nullptr;
  gpu::StripeDictionary const *stripe_dict = nullptr;
  gpu::DictionaryChunk *d_dict             = nullptr;
  gpu::StripeDictionary *d_stripe_dict     = nullptr;
};

void writer::impl::init_dictionaries(orc_column_view *columns,
                                     size_t num_rows,
                                     std::vector<int> const &str_col_ids,
                                     uint32_t *dict_data,
                                     uint32_t *dict_index,
                                     hostdevice_vector<gpu::DictionaryChunk> &dict,
                                     cudaStream_t stream)
{
  const size_t num_rowgroups = dict.size() / str_col_ids.size();

  // Setup per-rowgroup dictionary indexes for each dictionary-aware column
  for (size_t i = 0; i < str_col_ids.size(); ++i) {
    auto &str_column = columns[str_col_ids[i]];
    str_column.set_dict_stride(str_col_ids.size());
    str_column.attach_dict_chunk(dict.host_ptr(), dict.device_ptr());

    for (size_t g = 0; g < num_rowgroups; g++) {
      auto *ck              = &dict[g * str_col_ids.size() + i];
      ck->valid_map_base    = str_column.nulls();
      ck->column_data_base  = str_column.data();
      ck->dict_data         = dict_data + i * num_rows + g * row_index_stride_;
      ck->dict_index        = dict_index + i * num_rows;  // Indexed by abs row
      ck->start_row         = g * row_index_stride_;
      ck->num_rows          = std::min<uint32_t>(row_index_stride_,
                                        std::max<int>(str_column.data_count() - ck->start_row, 0));
      ck->num_strings       = 0;
      ck->string_char_count = 0;
      ck->num_dict_strings  = 0;
      ck->dict_char_count   = 0;
    }
  }

  CUDA_TRY(cudaMemcpyAsync(
    dict.device_ptr(), dict.host_ptr(), dict.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(
    gpu::InitDictionaryIndices(dict.device_ptr(), str_col_ids.size(), num_rowgroups, stream));
  CUDA_TRY(cudaMemcpyAsync(
    dict.host_ptr(), dict.device_ptr(), dict.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

void writer::impl::build_dictionaries(orc_column_view *columns,
                                      size_t num_rows,
                                      std::vector<int> const &str_col_ids,
                                      std::vector<uint32_t> const &stripe_list,
                                      hostdevice_vector<gpu::DictionaryChunk> const &dict,
                                      uint32_t *dict_index,
                                      hostdevice_vector<gpu::StripeDictionary> &stripe_dict,
                                      cudaStream_t stream)
{
  const auto num_rowgroups = dict.size() / str_col_ids.size();

  for (size_t i = 0; i < str_col_ids.size(); i++) {
    size_t direct_cost = 0, dict_cost = 0;
    auto &str_column = columns[str_col_ids[i]];
    str_column.attach_stripe_dict(stripe_dict.host_ptr(), stripe_dict.device_ptr());

    for (size_t j = 0, g = 0; j < stripe_list.size(); j++) {
      const auto num_chunks = stripe_list[j];
      auto *sd              = &stripe_dict[j * str_col_ids.size() + i];
      sd->column_data_base  = str_column.host_dict_chunk(0)->column_data_base;
      sd->dict_data         = str_column.host_dict_chunk(g)->dict_data;
      sd->dict_index        = dict_index + i * num_rows;  // Indexed by abs row
      sd->column_id         = str_col_ids[i];
      sd->start_chunk       = (uint32_t)g;
      sd->num_chunks        = num_chunks;
      sd->num_strings       = 0;
      sd->dict_char_count   = 0;
      for (size_t k = g; k < g + num_chunks; k++) {
        const auto &dt = dict[k * str_col_ids.size() + i];
        sd->num_strings += dt.num_dict_strings;
        direct_cost += dt.string_char_count;
        dict_cost += dt.dict_char_count + dt.num_dict_strings;
      }

      g += num_chunks;
    }

    // Early disable of dictionary if it doesn't look good at the chunk level
    if (enable_dictionary_ && dict_cost >= direct_cost) {
      for (size_t j = 0; j < stripe_list.size(); j++) {
        stripe_dict[j * str_col_ids.size() + i].dict_data = nullptr;
      }
    }
  }

  CUDA_TRY(cudaMemcpyAsync(stripe_dict.device_ptr(),
                           stripe_dict.host_ptr(),
                           stripe_dict.memory_size(),
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::BuildStripeDictionaries(stripe_dict.device_ptr(),
                                        stripe_dict.host_ptr(),
                                        dict.device_ptr(),
                                        stripe_list.size(),
                                        num_rowgroups,
                                        str_col_ids.size(),
                                        stream));
  CUDA_TRY(cudaMemcpyAsync(stripe_dict.host_ptr(),
                           stripe_dict.device_ptr(),
                           stripe_dict.memory_size(),
                           cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

std::vector<Stream> writer::impl::gather_streams(orc_column_view *columns,
                                                 size_t num_columns,
                                                 size_t num_rows,
                                                 std::vector<uint32_t> const &stripe_list,
                                                 std::vector<int32_t> &strm_ids,
                                                 const orc_chunked_state &state)
{
  // First n + 1 streams are row index streams, including 'column 0'
  std::vector<Stream> streams;
  streams.resize(num_columns + 1);
  streams[0].column = 0;
  streams[0].kind   = ROW_INDEX;
  streams[0].length = 0;

  for (size_t i = 0; i < num_columns; ++i) {
    TypeKind kind                    = columns[i].orc_kind();
    StreamKind data_kind             = DATA;
    StreamKind data2_kind            = LENGTH;
    ColumnEncodingKind encoding_kind = DIRECT;

    int64_t present_stream_size = 0;
    int64_t data_stream_size    = 0;
    int64_t data2_stream_size   = 0;
    int64_t dict_stream_size    = 0;
    bool is_nullable;

    if (state.single_write_mode) {
      is_nullable = (columns[i].nullable() || columns[i].data_count() < num_rows);
    } else {
      is_nullable = (i < state.user_metadata_with_nullability.column_nullable.size())
                      ? state.user_metadata_with_nullability.column_nullable[i]
                      : true;
    }
    if (is_nullable) {
      present_stream_size = ((row_index_stride_ + 7) >> 3);
      present_stream_size += (present_stream_size + 0x7f) >> 7;
    }

    switch (kind) {
      case TypeKind::BOOLEAN:
        data_stream_size = div_rowgroups_by<int64_t>(1024) * (128 + 1);
        encoding_kind    = DIRECT;
        break;
      case TypeKind::BYTE:
        data_stream_size = div_rowgroups_by<int64_t>(128) * (128 + 1);
        encoding_kind    = DIRECT;
        break;
      case TypeKind::SHORT:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 2 + 2);
        encoding_kind    = DIRECT_V2;
        break;
      case TypeKind::FLOAT:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size = (columns[i].null_count() != 0)
                             ? div_rowgroups_by<int64_t>(512) * (512 * 4 + 2)
                             : INT64_C(-1);
        encoding_kind = DIRECT;
        break;
      case TypeKind::INT:
      case TypeKind::DATE:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
        encoding_kind    = DIRECT_V2;
        break;
      case TypeKind::DOUBLE:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size = (columns[i].null_count() != 0)
                             ? div_rowgroups_by<int64_t>(512) * (512 * 8 + 2)
                             : INT64_C(-1);
        encoding_kind = DIRECT;
        break;
      case TypeKind::LONG:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 8 + 2);
        encoding_kind    = DIRECT_V2;
        break;
      case TypeKind::STRING: {
        bool enable_dict           = enable_dictionary_;
        size_t direct_data_size    = 0;
        size_t dict_data_size      = 0;
        size_t dict_strings        = 0;
        size_t dict_lengths_div512 = 0;
        for (size_t stripe = 0, g = 0; stripe < stripe_list.size(); stripe++) {
          const auto sd = columns[i].host_stripe_dict(stripe);
          enable_dict   = (enable_dict && sd->dict_data != nullptr);
          if (enable_dict) {
            dict_strings += sd->num_strings;
            dict_lengths_div512 += (sd->num_strings + 0x1ff) >> 9;
            dict_data_size += sd->dict_char_count;
          }

          for (uint32_t k = 0; k < stripe_list[stripe]; k++, g++) {
            direct_data_size += columns[i].host_dict_chunk(g)->string_char_count;
          }
        }
        if (enable_dict) {
          uint32_t dict_bits = 0;
          for (dict_bits = 1; dict_bits < 32; dict_bits <<= 1) {
            if (dict_strings <= (1ull << dict_bits)) break;
          }
          const auto valid_count = columns[i].data_count() - columns[i].null_count();
          dict_data_size += (dict_bits * valid_count + 7) >> 3;
        }

        // Decide between direct or dictionary encoding
        if (enable_dict && dict_data_size < direct_data_size) {
          data_stream_size  = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
          data2_stream_size = dict_lengths_div512 * (512 * 4 + 2);
          dict_stream_size  = std::max<size_t>(dict_data_size, 1);
          encoding_kind     = DICTIONARY_V2;
        } else {
          data_stream_size  = std::max<size_t>(direct_data_size, 1);
          data2_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
          encoding_kind     = DIRECT_V2;
        }
        break;
      }
      case TypeKind::TIMESTAMP:
        data_stream_size  = ((row_index_stride_ + 0x1ff) >> 9) * (512 * 4 + 2);
        data2_stream_size = data_stream_size;
        data2_kind        = SECONDARY;
        encoding_kind     = DIRECT_V2;
        break;
      default: CUDF_FAIL("Unsupported ORC type kind");
    }

    // Initialize the column's metadata
    columns[i].set_orc_encoding(encoding_kind);

    // Initialize the column's index stream
    const auto id      = static_cast<uint32_t>(1 + i);
    streams[id].column = id;
    streams[id].kind   = ROW_INDEX;
    streams[id].length = 0;

    // Initialize the column's data stream(s)
    const auto base = i * gpu::CI_NUM_STREAMS;
    if (present_stream_size != 0) {
      auto len                         = static_cast<uint64_t>(present_stream_size);
      strm_ids[base + gpu::CI_PRESENT] = streams.size();
      streams.push_back(orc::Stream{PRESENT, id, len});
    }
    if (data_stream_size != 0) {
      auto len                      = static_cast<uint64_t>(std::max<int64_t>(data_stream_size, 0));
      strm_ids[base + gpu::CI_DATA] = streams.size();
      streams.push_back(orc::Stream{data_kind, id, len});
    }
    if (data2_stream_size != 0) {
      auto len = static_cast<uint64_t>(std::max<int64_t>(data2_stream_size, 0));
      strm_ids[base + gpu::CI_DATA2] = streams.size();
      streams.push_back(orc::Stream{data2_kind, id, len});
    }
    if (dict_stream_size != 0) {
      auto len                            = static_cast<uint64_t>(dict_stream_size);
      strm_ids[base + gpu::CI_DICTIONARY] = streams.size();
      streams.push_back(orc::Stream{DICTIONARY_DATA, id, len});
    }
  }

  return streams;
}

rmm::device_buffer writer::impl::encode_columns(orc_column_view *columns,
                                                size_t num_columns,
                                                size_t num_rows,
                                                size_t num_rowgroups,
                                                std::vector<int> const &str_col_ids,
                                                std::vector<uint32_t> const &stripe_list,
                                                std::vector<Stream> const &streams,
                                                std::vector<int32_t> const &strm_ids,
                                                hostdevice_vector<gpu::EncChunk> &chunks,
                                                cudaStream_t stream)
{
  // Allocate combined buffer for RLE data and string data output
  std::vector<size_t> strm_offsets(streams.size());
  size_t str_data_size = 0;
  auto output          = [&]() {
    size_t rle_data_size = 0;
    for (size_t i = 0; i < streams.size(); ++i) {
      const auto &stream = streams[i];
      const auto &column = columns[stream.column - 1];

      if (((stream.kind == DICTIONARY_DATA || stream.kind == LENGTH) &&
           (column.orc_encoding() == DICTIONARY_V2)) ||
          ((stream.kind == DATA) &&
           (column.orc_kind() == TypeKind::STRING && column.orc_encoding() == DIRECT_V2))) {
        strm_offsets[i] = str_data_size;
        str_data_size += stream.length;
      } else {
        strm_offsets[i] = rle_data_size;
        rle_data_size += (stream.length * num_rowgroups + 7) & ~7;
      }
    }
    str_data_size = (str_data_size + 7) & ~7;

    return rmm::device_buffer(rle_data_size + str_data_size, stream);
  }();
  auto dst_base = static_cast<uint8_t *>(output.data());

  // Initialize column chunks' descriptions
  size_t stripe_start = 0;
  size_t stripe_id    = 0;
  for (size_t j = 0; j < num_rowgroups; j++) {
    for (size_t i = 0; i < num_columns; i++) {
      auto *ck          = &chunks[j * num_columns + i];
      ck->start_row     = (j * row_index_stride_);
      ck->num_rows      = std::min<uint32_t>(row_index_stride_, num_rows - ck->start_row);
      ck->valid_rows    = columns[i].data_count();
      ck->encoding_kind = columns[i].orc_encoding();
      ck->type_kind     = columns[i].orc_kind();
      if (ck->type_kind == TypeKind::STRING) {
        ck->valid_map_base   = columns[i].nulls();
        ck->column_data_base = (ck->encoding_kind == DICTIONARY_V2)
                                 ? columns[i].host_stripe_dict(stripe_id)->dict_index
                                 : columns[i].data();
        ck->dtype_len = 1;
      } else {
        ck->valid_map_base   = columns[i].nulls();
        ck->column_data_base = columns[i].data();
        ck->dtype_len        = columns[i].type_width();
      }
      ck->scale = columns[i].clockscale();

      for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
        const auto strm_id = strm_ids[i * gpu::CI_NUM_STREAMS + k];

        ck->strm_id[k] = strm_id;
        if (strm_id >= 0) {
          if ((k == gpu::CI_DICTIONARY) ||
              (k == gpu::CI_DATA2 && ck->encoding_kind == DICTIONARY_V2)) {
            if (j == stripe_start) {
              const int32_t dict_stride = columns[i].get_dict_stride();
              const auto stripe         = columns[i].host_stripe_dict(stripe_id);
              ck->strm_len[k]           = (k == gpu::CI_DICTIONARY)
                                  ? stripe->dict_char_count
                                  : (((stripe->num_strings + 0x1ff) >> 9) * (512 * 4 + 2));
              if (stripe_id == 0) {
                ck->streams[k] = dst_base + strm_offsets[strm_id];
              } else {
                const auto *ck_up = &chunks[stripe[-dict_stride].start_chunk * num_columns + i];
                ck->streams[k]    = ck_up->streams[k] + ck_up->strm_len[k];
              }
            } else {
              ck->strm_len[k] = 0;
              ck->streams[k]  = ck[-num_columns].streams[k];
            }
          } else if (k == gpu::CI_DATA && ck->type_kind == TypeKind::STRING &&
                     ck->encoding_kind == DIRECT_V2) {
            ck->strm_len[k] = columns[i].host_dict_chunk(j)->string_char_count;
            ck->streams[k]  = (j == 0)
                               ? dst_base + strm_offsets[strm_id]
                               : (ck[-num_columns].streams[k] + ck[-num_columns].strm_len[k]);
          } else if (k == gpu::CI_DATA && streams[strm_id].length == 0 &&
                     (ck->type_kind == DOUBLE || ck->type_kind == FLOAT)) {
            // Pass-through
            ck->strm_len[k] = ck->num_rows * ck->dtype_len;
            ck->streams[k]  = nullptr;
          } else {
            ck->strm_len[k] = streams[strm_id].length;
            ck->streams[k] =
              dst_base + str_data_size + strm_offsets[strm_id] + streams[strm_id].length * j;
          }
        } else {
          ck->strm_len[k] = 0;
          ck->streams[k]  = nullptr;
        }
      }
    }

    // Track the current stripe this rowgroup chunk belongs
    if (j + 1 == stripe_start + stripe_list[stripe_id]) {
      stripe_start = j + 1;
      stripe_id++;
    }
  }

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  if (!str_col_ids.empty()) {
    auto d_stripe_dict = columns[str_col_ids[0]].device_stripe_dict();
    CUDA_TRY(gpu::EncodeStripeDictionaries(d_stripe_dict,
                                           chunks.device_ptr(),
                                           str_col_ids.size(),
                                           num_columns,
                                           stripe_list.size(),
                                           stream));
  }
  CUDA_TRY(gpu::EncodeOrcColumnData(chunks.device_ptr(), num_columns, num_rowgroups, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  return output;
}

std::vector<StripeInformation> writer::impl::gather_stripes(
  size_t num_columns,
  size_t num_rows,
  size_t num_index_streams,
  size_t num_data_streams,
  std::vector<uint32_t> const &stripe_list,
  hostdevice_vector<gpu::EncChunk> &chunks,
  hostdevice_vector<gpu::StripeStream> &strm_desc,
  cudaStream_t stream)
{
  std::vector<StripeInformation> stripes(stripe_list.size());
  size_t group        = 0;
  size_t stripe_start = 0;
  for (size_t s = 0; s < stripe_list.size(); s++) {
    size_t stripe_group_end = group + stripe_list[s];

    for (size_t i = 0; i < num_columns; i++) {
      const auto *ck = &chunks[group * num_columns + i];

      // Assign stream data of column data stream(s)
      for (int k = 0; k < gpu::CI_INDEX; k++) {
        const auto stream_id = ck->strm_id[k];
        if (stream_id != -1) {
          auto *ss           = &strm_desc[s * num_data_streams + stream_id - num_index_streams];
          ss->stream_size    = 0;
          ss->first_chunk_id = (group * num_columns + i);
          ss->num_chunks     = (stripe_group_end - group);
          ss->column_id      = i;
          ss->stream_type    = k;
        }
      }
    }

    group                   = stripe_group_end;
    size_t stripe_end       = std::min(group * row_index_stride_, num_rows);
    stripes[s].numberOfRows = stripe_end - stripe_start;
    stripe_start            = stripe_end;
  }

  CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(),
                           strm_desc.host_ptr(),
                           strm_desc.memory_size(),
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::CompactOrcDataStreams(
    strm_desc.device_ptr(), chunks.device_ptr(), strm_desc.size(), num_columns, stream));
  CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(),
                           strm_desc.device_ptr(),
                           strm_desc.memory_size(),
                           cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaMemcpyAsync(
    chunks.host_ptr(), chunks.device_ptr(), chunks.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  return stripes;
}

std::vector<std::vector<uint8_t>> writer::impl::gather_statistic_blobs(
  orc_column_view const *columns,
  size_t num_columns,
  size_t num_rows,
  size_t num_rowgroups,
  std::vector<uint32_t> const &stripe_list,
  std::vector<StripeInformation> const &stripes,
  hostdevice_vector<gpu::EncChunk> &chunks,
  cudaStream_t stream)
{
  size_t num_stat_blobs = (1 + stripe_list.size()) * num_columns;
  size_t num_chunks     = chunks.size();
  std::vector<std::vector<uint8_t>> stat_blobs(num_stat_blobs);
  hostdevice_vector<stats_column_desc> stat_desc(num_columns);
  hostdevice_vector<statistics_merge_group> stat_merge(num_stat_blobs);
  rmm::device_vector<statistics_chunk> stat_chunks(num_chunks + num_stat_blobs);
  rmm::device_vector<statistics_group> stat_groups(num_chunks);

  for (size_t i = 0; i < num_columns; i++) {
    stats_column_desc *desc = &stat_desc[i];
    switch (columns[i].orc_kind()) {
      case TypeKind::BYTE: desc->stats_dtype = dtype_int8; break;
      case TypeKind::SHORT: desc->stats_dtype = dtype_int16; break;
      case TypeKind::INT: desc->stats_dtype = dtype_int32; break;
      case TypeKind::LONG: desc->stats_dtype = dtype_int64; break;
      case TypeKind::FLOAT: desc->stats_dtype = dtype_float32; break;
      case TypeKind::DOUBLE: desc->stats_dtype = dtype_float64; break;
      case TypeKind::BOOLEAN: desc->stats_dtype = dtype_bool; break;
      case TypeKind::DATE: desc->stats_dtype = dtype_int32; break;
      case TypeKind::TIMESTAMP: desc->stats_dtype = dtype_timestamp64; break;
      case TypeKind::STRING: desc->stats_dtype = dtype_string; break;
      default: desc->stats_dtype = dtype_none; break;
    }
    desc->num_rows         = columns[i].data_count();
    desc->valid_map_base   = columns[i].nulls();
    desc->column_data_base = columns[i].data();
    if (desc->stats_dtype == dtype_timestamp64) {
      // Timestamp statistics are in milliseconds
      switch (columns[i].clockscale()) {
        case 9: desc->ts_scale = 1000; break;
        case 6: desc->ts_scale = 0; break;
        case 3: desc->ts_scale = -1000; break;
        case 0: desc->ts_scale = -1000000; break;
        default: desc->ts_scale = 0; break;
      }
    } else {
      desc->ts_scale = 0;
    }
    for (size_t k = 0, c = 0; k < stripe_list.size(); k++) {
      statistics_merge_group *grp = &stat_merge[i * stripe_list.size() + k];
      grp->col                    = stat_desc.device_ptr(i);
      grp->start_chunk            = static_cast<uint32_t>(i * num_rowgroups + c);
      grp->num_chunks             = stripe_list[k];
      c += stripe_list[k];
    }
    statistics_merge_group *col_stats = &stat_merge[stripe_list.size() * num_columns + i];
    col_stats->col                    = stat_desc.device_ptr(i);
    col_stats->start_chunk            = static_cast<uint32_t>(i * stripe_list.size());
    col_stats->num_chunks             = static_cast<uint32_t>(stripe_list.size());
  }
  CUDA_TRY(cudaMemcpyAsync(stat_desc.device_ptr(),
                           stat_desc.host_ptr(),
                           stat_desc.memory_size(),
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(cudaMemcpyAsync(stat_merge.device_ptr(),
                           stat_merge.host_ptr(),
                           stat_merge.memory_size(),
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::orc_init_statistics_groups(stat_groups.data().get(),
                                           stat_desc.device_ptr(),
                                           num_columns,
                                           num_rowgroups,
                                           row_index_stride_,
                                           stream));
  CUDA_TRY(
    GatherColumnStatistics(stat_chunks.data().get(), stat_groups.data().get(), num_chunks, stream));
  CUDA_TRY(MergeColumnStatistics(stat_chunks.data().get() + num_chunks,
                                 stat_chunks.data().get(),
                                 stat_merge.device_ptr(),
                                 stripe_list.size() * num_columns,
                                 stream));
  CUDA_TRY(
    MergeColumnStatistics(stat_chunks.data().get() + num_chunks + stripe_list.size() * num_columns,
                          stat_chunks.data().get() + num_chunks,
                          stat_merge.device_ptr(stripe_list.size() * num_columns),
                          num_columns,
                          stream));
  CUDA_TRY(gpu::orc_init_statistics_buffersize(
    stat_merge.device_ptr(), stat_chunks.data().get() + num_chunks, num_stat_blobs, stream));
  CUDA_TRY(cudaMemcpyAsync(stat_merge.host_ptr(),
                           stat_merge.device_ptr(),
                           stat_merge.memory_size(),
                           cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  hostdevice_vector<uint8_t> blobs(stat_merge[num_stat_blobs - 1].start_chunk +
                                   stat_merge[num_stat_blobs - 1].num_chunks);
  CUDA_TRY(gpu::orc_encode_statistics(blobs.device_ptr(),
                                      stat_merge.device_ptr(),
                                      stat_chunks.data().get() + num_chunks,
                                      num_stat_blobs,
                                      stream));
  CUDA_TRY(cudaMemcpyAsync(stat_merge.host_ptr(),
                           stat_merge.device_ptr(),
                           stat_merge.memory_size(),
                           cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaMemcpyAsync(
    blobs.host_ptr(), blobs.device_ptr(), blobs.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < num_stat_blobs; i++) {
    const uint8_t *stat_begin = blobs.host_ptr(stat_merge[i].start_chunk);
    const uint8_t *stat_end   = stat_begin + stat_merge[i].num_chunks;
    stat_blobs[i].assign(stat_begin, stat_end);
  }

  return stat_blobs;
}

void writer::impl::write_index_stream(int32_t stripe_id,
                                      int32_t stream_id,
                                      orc_column_view *columns,
                                      size_t num_columns,
                                      size_t num_data_streams,
                                      size_t group,
                                      size_t groups_in_stripe,
                                      hostdevice_vector<gpu::EncChunk> const &chunks,
                                      hostdevice_vector<gpu::StripeStream> const &strm_desc,
                                      hostdevice_vector<gpu_inflate_status_s> const &comp_out,
                                      StripeInformation &stripe,
                                      std::vector<Stream> &streams,
                                      ProtobufWriter *pbw)
{
  // 0: position, 1: block position, 2: compressed position, 3: compressed size
  std::array<int32_t, 4> present;
  std::array<int32_t, 4> data;
  std::array<int32_t, 4> data2;
  auto kind = TypeKind::STRUCT;

  auto find_record = [=, &strm_desc](gpu::EncChunk const &chunk, gpu::StreamIndexType type) {
    std::array<int32_t, 4> record{-1, -1, -1, -1};
    if (chunk.strm_id[type] > 0) {
      record[0] = 0;
      if (compression_kind_ != NONE) {
        const auto *ss =
          &strm_desc[stripe_id * num_data_streams + chunk.strm_id[type] - (num_columns + 1)];
        record[1] = ss->first_block;
        record[2] = 0;
        record[3] = ss->stream_size;
      }
    }
    return record;
  };
  auto scan_record = [=, &comp_out](gpu::EncChunk const &chunk,
                                    gpu::StreamIndexType type,
                                    std::array<int32_t, 4> &record) {
    if (record[0] >= 0) {
      record[0] += chunk.strm_len[type];
      while ((record[1] >= 0) && (static_cast<size_t>(record[0]) >= compression_blocksize_) &&
             (record[2] + 3 + comp_out[record[1]].bytes_written < static_cast<size_t>(record[3]))) {
        record[0] -= compression_blocksize_;
        record[2] += 3 + comp_out[record[1]].bytes_written;
        record[1] += 1;
      }
    }
  };

  // TBD: Not sure we need an empty index stream for column 0
  if (stream_id != 0) {
    const auto &ck = chunks[stream_id - 1];
    present        = find_record(ck, gpu::CI_PRESENT);
    data           = find_record(ck, gpu::CI_DATA);
    data2          = find_record(ck, gpu::CI_DATA2);

    // Change string dictionary to int from index point of view
    kind = columns[stream_id - 1].orc_kind();
    if (kind == TypeKind::STRING && columns[stream_id - 1].orc_encoding() == DICTIONARY_V2) {
      kind = TypeKind::INT;
    }
  }

  buffer_.resize((compression_kind_ != NONE) ? 3 : 0);

  // Add row index entries
  for (size_t g = group; g < group + groups_in_stripe; g++) {
    pbw->put_row_index_entry(present[2], present[0], data[2], data[0], data2[2], data2[0], kind);

    if (stream_id != 0) {
      const auto &ck = chunks[g * num_columns + stream_id - 1];
      scan_record(ck, gpu::CI_PRESENT, present);
      scan_record(ck, gpu::CI_DATA, data);
      scan_record(ck, gpu::CI_DATA2, data2);
    }
  }

  streams[stream_id].length = buffer_.size();
  if (compression_kind_ != NONE) {
    uint32_t uncomp_ix_len = (uint32_t)(streams[stream_id].length - 3) * 2 + 1;
    buffer_[0]             = static_cast<uint8_t>(uncomp_ix_len >> 0);
    buffer_[1]             = static_cast<uint8_t>(uncomp_ix_len >> 8);
    buffer_[2]             = static_cast<uint8_t>(uncomp_ix_len >> 16);
  }
  out_sink_->host_write(buffer_.data(), buffer_.size());
  stripe.indexLength += buffer_.size();
}

void writer::impl::write_data_stream(gpu::StripeStream const &strm_desc,
                                     gpu::EncChunk const &chunk,
                                     uint8_t const *compressed_data,
                                     uint8_t *stream_out,
                                     StripeInformation &stripe,
                                     std::vector<Stream> &streams,
                                     cudaStream_t stream)
{
  const auto length                                    = strm_desc.stream_size;
  streams[chunk.strm_id[strm_desc.stream_type]].length = length;
  if (length != 0) {
    const auto *stream_in = (compression_kind_ == NONE) ? chunk.streams[strm_desc.stream_type]
                                                        : (compressed_data + strm_desc.bfr_offset);
    CUDA_TRY(cudaMemcpyAsync(stream_out, stream_in, length, cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    out_sink_->host_write(stream_out, length);
  }
  stripe.dataLength += length;
}

void writer::impl::add_uncompressed_block_headers(std::vector<uint8_t> &v)
{
  if (compression_kind_ != NONE) {
    size_t uncomp_len = v.size() - 3, pos = 0, block_len;
    while (uncomp_len > compression_blocksize_) {
      block_len  = compression_blocksize_ * 2 + 1;
      v[pos + 0] = static_cast<uint8_t>(block_len >> 0);
      v[pos + 1] = static_cast<uint8_t>(block_len >> 8);
      v[pos + 2] = static_cast<uint8_t>(block_len >> 16);
      pos += 3 + compression_blocksize_;
      v.insert(v.begin() + pos, 3, 0);
      uncomp_len -= compression_blocksize_;
    }
    block_len  = uncomp_len * 2 + 1;
    v[pos + 0] = static_cast<uint8_t>(block_len >> 0);
    v[pos + 1] = static_cast<uint8_t>(block_len >> 8);
    v[pos + 2] = static_cast<uint8_t>(block_len >> 16);
  }
}

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   orc_writer_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : compression_kind_(to_orc_compression(options.get_compression())),
    enable_statistics_(options.enable_statistics()),
    out_sink_(std::move(sink)),
    _mr(mr)
{
}

void writer::impl::write(table_view const &table,
                         const table_metadata *metadata,
                         cudaStream_t stream)
{
  orc_chunked_state state;
  state.user_metadata     = metadata;
  state.stream            = stream;
  state.single_write_mode = true;

  write_chunked_begin(state);
  write_chunk(table, state);
  write_chunked_end(state);
}

void writer::impl::write_chunked_begin(orc_chunked_state &state)
{
  // Write file header
  out_sink_->host_write(MAGIC, std::strlen(MAGIC));
}

void writer::impl::write_chunk(table_view const &table, orc_chunked_state &state)
{
  size_type num_columns = table.num_columns();
  size_type num_rows    = 0;

  // Mapping of string columns for quick look-up
  std::vector<int> str_col_ids;

  if (state.user_metadata_with_nullability.column_nullable.size() > 0) {
    CUDF_EXPECTS(state.user_metadata_with_nullability.column_nullable.size() ==
                   static_cast<size_t>(num_columns),
                 "When passing values in user_metadata_with_nullability, data for all columns must "
                 "be specified");
  }

  // Wrapper around cudf columns to attach ORC-specific type info
  std::vector<orc_column_view> orc_columns;
  orc_columns.reserve(num_columns);  // Avoids unnecessary re-allocation
  for (auto it = table.begin(); it < table.end(); ++it) {
    const auto col            = *it;
    const auto current_id     = orc_columns.size();
    const auto current_str_id = str_col_ids.size();

    num_rows = std::max<uint32_t>(num_rows, col.size());
    orc_columns.emplace_back(current_id, current_str_id, col, state.user_metadata, state.stream);
    if (orc_columns.back().is_string()) { str_col_ids.push_back(current_id); }
  }

  rmm::device_vector<uint32_t> dict_index(str_col_ids.size() * num_rows);
  rmm::device_vector<uint32_t> dict_data(str_col_ids.size() * num_rows);

  // Build per-column dictionary indices
  const auto num_rowgroups   = div_by_rowgroups<size_t>(num_rows);
  const auto num_dict_chunks = num_rowgroups * str_col_ids.size();
  hostdevice_vector<gpu::DictionaryChunk> dict(num_dict_chunks);
  if (str_col_ids.size() != 0) {
    init_dictionaries(orc_columns.data(),
                      num_rows,
                      str_col_ids,
                      dict_data.data().get(),
                      dict_index.data().get(),
                      dict,
                      state.stream);
  }

  // Decide stripe boundaries early on, based on uncompressed size
  std::vector<uint32_t> stripe_list;
  for (size_t g = 0, stripe_start = 0, stripe_size = 0; g < num_rowgroups; g++) {
    size_t rowgroup_size = 0;
    for (int i = 0; i < num_columns; i++) {
      if (orc_columns[i].is_string()) {
        const auto dt = orc_columns[i].host_dict_chunk(g);
        rowgroup_size += 1 * row_index_stride_;
        rowgroup_size += dt->string_char_count;
      } else {
        rowgroup_size += orc_columns[i].type_width() * row_index_stride_;
      }
    }

    // Apply rows per stripe limit to limit string dictionaries
    const size_t max_stripe_rows = !str_col_ids.empty() ? 1000000 : 5000000;
    if ((g > stripe_start) && (stripe_size + rowgroup_size > max_stripe_size_ ||
                               (g + 1 - stripe_start) * row_index_stride_ > max_stripe_rows)) {
      stripe_list.push_back(g - stripe_start);
      stripe_start = g;
      stripe_size  = 0;
    }
    stripe_size += rowgroup_size;
    if (g + 1 == num_rowgroups) { stripe_list.push_back(num_rowgroups - stripe_start); }
  }

  // Build stripe-level dictionaries
  const auto num_stripe_dict = stripe_list.size() * str_col_ids.size();
  hostdevice_vector<gpu::StripeDictionary> stripe_dict(num_stripe_dict);
  if (str_col_ids.size() != 0) {
    build_dictionaries(orc_columns.data(),
                       num_rows,
                       str_col_ids,
                       stripe_list,
                       dict,
                       dict_index.data().get(),
                       stripe_dict,
                       state.stream);
  }

  // Initialize streams
  std::vector<int32_t> strm_ids(num_columns * gpu::CI_NUM_STREAMS, -1);
  auto streams =
    gather_streams(orc_columns.data(), orc_columns.size(), num_rows, stripe_list, strm_ids, state);

  // Encode column data chunks
  const auto num_chunks = num_rowgroups * num_columns;
  hostdevice_vector<gpu::EncChunk> chunks(num_chunks);
  auto output = encode_columns(orc_columns.data(),
                               num_columns,
                               num_rows,
                               num_rowgroups,
                               str_col_ids,
                               stripe_list,
                               streams,
                               strm_ids,
                               chunks,
                               state.stream);

  // Assemble individual disparate column chunks into contiguous data streams
  const auto num_index_streams  = (num_columns + 1);
  const auto num_data_streams   = streams.size() - num_index_streams;
  const auto num_stripe_streams = stripe_list.size() * num_data_streams;
  hostdevice_vector<gpu::StripeStream> strm_desc(num_stripe_streams);
  auto stripes = gather_stripes(num_columns,
                                num_rows,
                                num_index_streams,
                                num_data_streams,
                                stripe_list,
                                chunks,
                                strm_desc,
                                state.stream);

  // Gather column statistics
  std::vector<std::vector<uint8_t>> column_stats;
  if (enable_statistics_ && num_columns > 0 && num_rows > 0) {
    column_stats = gather_statistic_blobs(orc_columns.data(),
                                          num_columns,
                                          num_rows,
                                          num_rowgroups,
                                          stripe_list,
                                          stripes,
                                          chunks,
                                          state.stream);
  }

  // Allocate intermediate output stream buffer
  size_t compressed_bfr_size   = 0;
  size_t num_compressed_blocks = 0;
  auto stream_output           = [&]() {
    size_t max_stream_size = 0;

    for (size_t stripe_id = 0; stripe_id < stripe_list.size(); stripe_id++) {
      for (size_t i = 0; i < num_data_streams; i++) {
        gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + i];
        size_t stream_size    = ss->stream_size;
        if (compression_kind_ != NONE) {
          ss->first_block = num_compressed_blocks;
          ss->bfr_offset  = compressed_bfr_size;

          auto num_blocks = std::max<uint32_t>(
            (stream_size + compression_blocksize_ - 1) / compression_blocksize_, 1);
          stream_size += num_blocks * 3;
          num_compressed_blocks += num_blocks;
          compressed_bfr_size += stream_size;
        }
        max_stream_size = std::max(max_stream_size, stream_size);
      }
    }

    return pinned_buffer<uint8_t>{[](size_t size) {
                                    uint8_t *ptr = nullptr;
                                    CUDA_TRY(cudaMallocHost(&ptr, size));
                                    return ptr;
                                  }(max_stream_size),
                                  cudaFreeHost};
  }();

  // Compress the data streams
  rmm::device_buffer compressed_data(compressed_bfr_size, state.stream);
  hostdevice_vector<gpu_inflate_status_s> comp_out(num_compressed_blocks);
  hostdevice_vector<gpu_inflate_input_s> comp_in(num_compressed_blocks);
  if (compression_kind_ != NONE) {
    CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(),
                             strm_desc.host_ptr(),
                             strm_desc.memory_size(),
                             cudaMemcpyHostToDevice,
                             state.stream));
    CUDA_TRY(gpu::CompressOrcDataStreams(static_cast<uint8_t *>(compressed_data.data()),
                                         strm_desc.device_ptr(),
                                         chunks.device_ptr(),
                                         comp_in.device_ptr(),
                                         comp_out.device_ptr(),
                                         num_stripe_streams,
                                         num_compressed_blocks,
                                         compression_kind_,
                                         compression_blocksize_,
                                         state.stream));
    CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(),
                             strm_desc.device_ptr(),
                             strm_desc.memory_size(),
                             cudaMemcpyDeviceToHost,
                             state.stream));
    CUDA_TRY(cudaMemcpyAsync(comp_out.host_ptr(),
                             comp_out.device_ptr(),
                             comp_out.memory_size(),
                             cudaMemcpyDeviceToHost,
                             state.stream));
    CUDA_TRY(cudaStreamSynchronize(state.stream));
  }

  ProtobufWriter pbw_(&buffer_);

  // Write stripes
  size_t group = 0;
  for (size_t stripe_id = 0; stripe_id < stripes.size(); stripe_id++) {
    auto groups_in_stripe     = div_by_rowgroups(stripes[stripe_id].numberOfRows);
    stripes[stripe_id].offset = out_sink_->bytes_written();

    // Column (skippable) index streams appear at the start of the stripe
    stripes[stripe_id].indexLength = 0;
    for (size_t col_id = 0; col_id <= (size_t)num_columns; col_id++) {
      write_index_stream(stripe_id,
                         col_id,
                         orc_columns.data(),
                         num_columns,
                         num_data_streams,
                         group,
                         groups_in_stripe,
                         chunks,
                         strm_desc,
                         comp_out,
                         stripes[stripe_id],
                         streams,
                         &pbw_);
    }

    // Column data consisting one or more separate streams
    stripes[stripe_id].dataLength = 0;
    for (size_t i = 0; i < num_data_streams; i++) {
      const auto &ss = strm_desc[stripe_id * num_data_streams + i];
      const auto &ck = chunks[group * num_columns + ss.column_id];

      write_data_stream(ss,
                        ck,
                        static_cast<uint8_t *>(compressed_data.data()),
                        stream_output.get(),
                        stripes[stripe_id],
                        streams,
                        state.stream);
    }

    // Write stripefooter consisting of stream information
    StripeFooter sf;
    sf.streams = streams;
    sf.columns.resize(num_columns + 1);
    sf.columns[0].kind           = DIRECT;
    sf.columns[0].dictionarySize = 0;
    for (size_t i = 1; i < sf.columns.size(); ++i) {
      sf.columns[i].kind           = orc_columns[i - 1].orc_encoding();
      sf.columns[i].dictionarySize = (sf.columns[i].kind == DICTIONARY_V2)
                                       ? orc_columns[i - 1].host_stripe_dict(stripe_id)->num_strings
                                       : 0;
      if (orc_columns[i - 1].orc_kind() == TIMESTAMP) { sf.writerTimezone = "UTC"; }
    }
    buffer_.resize((compression_kind_ != NONE) ? 3 : 0);
    pbw_.write(&sf);
    stripes[stripe_id].footerLength = buffer_.size();
    if (compression_kind_ != NONE) {
      uint32_t uncomp_sf_len = (stripes[stripe_id].footerLength - 3) * 2 + 1;
      buffer_[0]             = static_cast<uint8_t>(uncomp_sf_len >> 0);
      buffer_[1]             = static_cast<uint8_t>(uncomp_sf_len >> 8);
      buffer_[2]             = static_cast<uint8_t>(uncomp_sf_len >> 16);
    }
    out_sink_->host_write(buffer_.data(), buffer_.size());

    group += groups_in_stripe;
  }

  if (column_stats.size() != 0) {
    // File-level statistics
    // NOTE: Excluded from chunked write mode to avoid the need for merging stats across calls
    if (state.single_write_mode) {
      state.ff.statistics.resize(1 + num_columns);
      // First entry contains total number of rows
      buffer_.resize(0);
      pbw_.putb(1 * 8 + PB_TYPE_VARINT);
      pbw_.put_uint(num_rows);
      state.ff.statistics[0] = std::move(buffer_);
      for (int i = 0; i < num_columns; i++) {
        size_t idx = stripe_list.size() * num_columns + i;
        if (idx < column_stats.size()) {
          state.ff.statistics[1 + i] = std::move(column_stats[idx]);
        }
      }
    }
    // Stripe-level statistics
    size_t first_stripe = state.md.stripeStats.size();
    state.md.stripeStats.resize(first_stripe + stripe_list.size());
    for (size_t stripe_id = 0; stripe_id < stripe_list.size(); stripe_id++) {
      state.md.stripeStats[first_stripe + stripe_id].colStats.resize(1 + num_columns);
      buffer_.resize(0);
      pbw_.putb(1 * 8 + PB_TYPE_VARINT);
      pbw_.put_uint(stripes[stripe_id].numberOfRows);
      state.md.stripeStats[first_stripe + stripe_id].colStats[0] = std::move(buffer_);
      for (int i = 0; i < num_columns; i++) {
        size_t idx = stripe_list.size() * i + stripe_id;
        if (idx < column_stats.size()) {
          state.md.stripeStats[first_stripe + stripe_id].colStats[1 + i] =
            std::move(column_stats[idx]);
        }
      }
    }
  }
  if (state.ff.headerLength == 0) {
    // First call
    state.ff.headerLength   = std::strlen(MAGIC);
    state.ff.rowIndexStride = row_index_stride_;
    state.ff.types.resize(1 + num_columns);
    state.ff.types[0].kind = STRUCT;
    state.ff.types[0].subtypes.resize(num_columns);
    state.ff.types[0].fieldNames.resize(num_columns);
    for (int i = 0; i < num_columns; ++i) {
      state.ff.types[1 + i].kind      = orc_columns[i].orc_kind();
      state.ff.types[0].subtypes[i]   = 1 + i;
      state.ff.types[0].fieldNames[i] = orc_columns[i].orc_name();
    }
  } else {
    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(state.ff.types.size() == 1 + orc_columns.size(),
                 "Mismatch in table structure between multiple calls to write_chunk");
    for (auto i = 0; i < num_columns; i++) {
      CUDF_EXPECTS(state.ff.types[1 + i].kind == orc_columns[i].orc_kind(),
                   "Mismatch in column types between multiple calls to write_chunk");
    }
  }
  state.ff.stripes.insert(state.ff.stripes.end(),
                          std::make_move_iterator(stripes.begin()),
                          std::make_move_iterator(stripes.end()));
  state.ff.numberOfRows += num_rows;
}

void writer::impl::write_chunked_end(orc_chunked_state &state)
{
  ProtobufWriter pbw_(&buffer_);
  PostScript ps;

  state.ff.contentLength = out_sink_->bytes_written();
  if (state.user_metadata) {
    for (auto it = state.user_metadata->user_data.begin();
         it != state.user_metadata->user_data.end();
         it++) {
      state.ff.metadata.push_back({it->first, it->second});
    }
  }
  // Write statistics metadata
  if (state.md.stripeStats.size() != 0) {
    buffer_.resize((compression_kind_ != NONE) ? 3 : 0);
    pbw_.write(&state.md);
    add_uncompressed_block_headers(buffer_);
    ps.metadataLength = buffer_.size();
    out_sink_->host_write(buffer_.data(), buffer_.size());
  } else {
    ps.metadataLength = 0;
  }
  buffer_.resize((compression_kind_ != NONE) ? 3 : 0);
  pbw_.write(&state.ff);
  add_uncompressed_block_headers(buffer_);

  // Write postscript metadata
  ps.footerLength         = buffer_.size();
  ps.compression          = compression_kind_;
  ps.compressionBlockSize = compression_blocksize_;
  ps.version              = {0, 12};
  ps.magic                = MAGIC;
  const auto ps_length    = static_cast<uint8_t>(pbw_.write(&ps));
  buffer_.push_back(ps_length);
  out_sink_->host_write(buffer_.data(), buffer_.size());
  out_sink_->flush();
}

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               orc_writer_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mr))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write(table_view const &table, const table_metadata *metadata, cudaStream_t stream)
{
  _impl->write(table, metadata, stream);
}

// Forward to implementation
void writer::write_chunked_begin(orc_chunked_state &state) { _impl->write_chunked_begin(state); }

// Forward to implementation
void writer::write_chunk(table_view const &table, orc_chunked_state &state)
{
  _impl->write_chunk(table, state);
}

// Forward to implementation
void writer::write_chunked_end(orc_chunked_state &state) { _impl->write_chunked_end(state); }

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
