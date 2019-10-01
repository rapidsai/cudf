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

#include "orc_writer_impl.hpp"

#include <cstring>

#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <io/utilities/wrapper_utils.hpp>

namespace {

template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

}  // namespace

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Function that translates GDF compression to ORC compression
 **/
constexpr CompressionKind to_orc_compression(compression_type compression) {
  switch (compression) {
    case compression_type::snappy:
      return SNAPPY;
    case compression_type::none:
    default:
      return NONE;
  }
}

/**
 * @brief Function that translates GDF dtype to ORC datatype
 **/
constexpr TypeKind to_orc_type(gdf_dtype dtype) {
  switch (dtype) {
    case GDF_INT8:
      return BYTE;
    case GDF_INT16:
      return SHORT;
    case GDF_INT32:
      return INT;
    case GDF_INT64:
      return LONG;
    case GDF_FLOAT32:
      return FLOAT;
    case GDF_FLOAT64:
      return DOUBLE;
    case GDF_BOOL8:
      return BOOLEAN;
    case GDF_DATE32:
      return DATE;
    case GDF_DATE64:
    case GDF_TIMESTAMP:
      return TIMESTAMP;
    case GDF_CATEGORY:
      return INT;
    case GDF_STRING:
    case GDF_STRING_CATEGORY:
      return STRING;
    default:
      return INVALID_TYPE_KIND;
  }
}

/**
 * @brief Function that translates time unit to nanoscale multiple
 **/
template <typename T>
constexpr T to_clockscale(gdf_time_unit time_unit) {
  switch (time_unit) {
    case TIME_UNIT_s:
      return 9;
    case TIME_UNIT_ms:
      return 6;
    case TIME_UNIT_us:
      return 3;
    case TIME_UNIT_ns:
    default:
      return 0;
  }
}

/**
 * @brief Helper class that wraps a cudf column to add ORC-specific info
 **/
class orc_column {
 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   **/
  explicit orc_column(size_t id, const gdf_column *col) : id(id), col(col) {
    if (col->dtype == GDF_STRING) {
      auto *str = static_cast<NVStrings *>(col->data);
      index.resize(col->size);
      CUDF_EXPECTS(str->create_index(index.data()) == 0,
                   "Cannot retrieve NVStrings string pairs");
    } else if (col->dtype == GDF_STRING_CATEGORY) {
      auto *cat = static_cast<NVCategory *>(col->dtype_info.category);
      index.resize(col->size);
      CUDF_EXPECTS(cat->create_index(index.data()) == 0,
                   "Cannot retrieve NVCategory string pairs");
    }
  }

  /**
   * @brief Convenience operator to access the wrapped cudf column
   **/
  const gdf_column *operator->() const noexcept { return col; }

  void set_dict_stride(size_t stride) { dict_stride = stride; }
  auto get_dict_stride() { return dict_stride; }

  /**
   * @brief Function that associates an existing dictionary chunk allocation
   **/
  void attach_dict_chunk(gpu::DictionaryChunk *host_dict,
                         gpu::DictionaryChunk *dev_dict) {
    dict = host_dict;
    d_dict = dev_dict;
  }
  auto host_dict_chunk(size_t rowgroup) {
    assert(col->dtype == GDF_STRING || col->dtype == GDF_STRING_CATEGORY);
    return &dict[rowgroup * dict_stride + id];
  }
  auto device_dict_chunk() const { return d_dict; }

  /**
   * @brief Function that associates an existing stripe dictionary allocation
   **/
  void attach_stripe_dict(gpu::StripeDictionary *host_stripe_dict,
                          gpu::StripeDictionary *dev_stripe_dict) {
    stripe_dict = host_stripe_dict;
    d_stripe_dict = dev_stripe_dict;
  }
  auto host_stripe_dict(size_t stripe) const {
    assert(col->dtype == GDF_STRING || col->dtype == GDF_STRING_CATEGORY);
    return &stripe_dict[stripe * dict_stride + id];
  }
  auto device_stripe_dict() const { return d_stripe_dict; }

  void set_orc_kind(TypeKind t) { type_kind = t; }
  void set_orc_encoding(ColumnEncodingKind e) { encoding_kind = e; }
  const auto orc_data() const {
    return index.empty() ? col->data : index.data();
  }
  const auto orc_valid() const {
    return reinterpret_cast<uint32_t *>((col->null_count != 0) ? col->valid
                                                               : nullptr);
  }
  const auto orc_kind() const { return type_kind; }
  const auto orc_encoding() const { return encoding_kind; }
  const auto orc_name() const {
    return (col->col_name != nullptr) ? std::string(col->col_name)
                                      : "_col" + std::to_string(id);
  }

 private:
  size_t id = 0;

  // cudf-related members
  const gdf_column *col = nullptr;
  device_buffer<std::pair<const char *, size_t>> index;

  // ORC-related members
  TypeKind type_kind;
  ColumnEncodingKind encoding_kind;

  // String dictionary-related members
  size_t dict_stride = 0;
  const gpu::DictionaryChunk *dict = nullptr;
  const gpu::StripeDictionary *stripe_dict = nullptr;
  gpu::DictionaryChunk *d_dict = nullptr;
  gpu::StripeDictionary *d_stripe_dict = nullptr;
};

writer::Impl::Impl(std::string filepath, writer_options const &options) {
  compression_kind_ = to_orc_compression(options.compression);

  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::Impl::write(const cudf::table &table) {
  gdf_size_type num_columns = table.num_columns();
  gdf_size_type num_rows = 0;

  // Mapping of string columns for quick look-up
  std::vector<int> str_col_ids;

  // Wrapper around cudf columns to attach ORC-specific type info
  std::vector<orc_column> orc_columns;
  for (auto it = table.begin(); it < table.end(); ++it) {
    const auto col = *it;
    num_rows = std::max(num_rows, col->size);
    if (col->dtype == GDF_STRING || col->dtype == GDF_STRING_CATEGORY) {
      str_col_ids.push_back(orc_columns.size());
    }
    orc_columns.emplace_back(orc_columns.size(), col);
  }

  device_buffer<uint32_t> dict_index(str_col_ids.size() * num_rows);
  device_buffer<uint32_t> dict_data(str_col_ids.size() * num_rows);

  // Build per-column dictionary indices
  const auto num_rowgroups = div_by_rowgroups<size_t>(num_rows);
  const auto num_dict_chunks = num_rowgroups * str_col_ids.size();
  hostdevice_vector<gpu::DictionaryChunk> dict(num_dict_chunks);
  if (!str_col_ids.empty()) {
    init_dictionaries(orc_columns.data(), num_rows, str_col_ids, dict_data,
                      dict_index, dict);
  }

  // Decide stripe boundaries early on, based on uncompressed size
  std::vector<uint32_t> stripe_list;
  for (size_t g = 0, stripe_start = 0, stripe_size = 0; g < num_rowgroups;
       g++) {
    size_t rowgroup_size = 0;
    for (int i = 0; i < num_columns; i++) {
      if (orc_columns[i]->dtype == GDF_STRING ||
          orc_columns[i]->dtype == GDF_STRING_CATEGORY) {
        const auto dt = orc_columns[i].host_dict_chunk(g);
        rowgroup_size += 1 * row_index_stride_;
        rowgroup_size += dt->string_char_count;
      } else {
        rowgroup_size +=
            cudf::size_of(orc_columns[i]->dtype) * row_index_stride_;
      }
    }

    // Apply rows per stripe limit to limit string dictionaries
    const size_t max_stripe_rows = !str_col_ids.empty() ? 1000000 : 5000000;
    if ((g > stripe_start) &&
        (stripe_size + rowgroup_size > max_stripe_size_ ||
         (g + 1 - stripe_start) * row_index_stride_ > max_stripe_rows)) {
      stripe_list.push_back(g - stripe_start);
      stripe_start = g;
      stripe_size = 0;
    }
    stripe_size += rowgroup_size;
    if (g + 1 == num_rowgroups) {
      stripe_list.push_back(num_rowgroups - stripe_start);
    }
  }

  // Build stripe-level dictionaries
  const auto num_stripe_dict = stripe_list.size() * str_col_ids.size();
  hostdevice_vector<gpu::StripeDictionary> stripe_dict(num_stripe_dict);
  if (!str_col_ids.empty()) {
    build_dictionaries(orc_columns.data(), num_rows, str_col_ids, stripe_list,
                       dict, dict_index, stripe_dict);
  }

  // Initialize streams
  std::vector<int32_t> strm_ids(num_columns * gpu::CI_NUM_STREAMS, -1);
  auto streams = gather_streams(orc_columns.data(), orc_columns.size(),
                                num_rows, stripe_list, strm_ids);

  // Encode column data chunks
  const auto num_chunks = num_rowgroups * num_columns;
  hostdevice_vector<gpu::EncChunk> chunks(num_chunks);
  auto output =
      encode_columns(orc_columns.data(), num_columns, num_rows, num_rowgroups,
                     str_col_ids, stripe_list, streams, strm_ids, chunks);

  // Assemble individual desparate column chunks into contiguous data streams
  const auto num_index_streams = (num_columns + 1);
  const auto num_data_streams = streams.size() - num_index_streams;
  const auto num_stripe_streams = stripe_list.size() * num_data_streams;
  hostdevice_vector<gpu::StripeStream> strm_desc(num_stripe_streams);
  auto stripes =
      gather_stripes(num_columns, num_rows, num_index_streams, num_data_streams,
                     stripe_list, chunks, strm_desc);

  // Allocate intermediate output stream buffer
  size_t compressed_bfr_size = 0;
  size_t num_compressed_blocks = 0;
  auto stream_output = [&]() {
    size_t max_stream_size = 0;

    for (size_t stripe_id = 0; stripe_id < stripe_list.size(); stripe_id++) {
      for (size_t i = 0; i < num_data_streams; i++) {
        gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + i];
        size_t stream_size = ss->stream_size;
        if (compression_kind_ != NONE) {
          ss->first_block = num_compressed_blocks;
          ss->bfr_offset = compressed_bfr_size;

          auto num_blocks =
              std::max<uint32_t>((stream_size + compression_blocksize_ - 1) /
                                     compression_blocksize_,
                                 1);
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
  device_buffer<uint8_t> compressed_data(compressed_bfr_size);
  hostdevice_vector<gpu_inflate_status_s> comp_out(num_compressed_blocks);
  hostdevice_vector<gpu_inflate_input_s> comp_in(num_compressed_blocks);
  if (compression_kind_ != NONE) {
    CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(), strm_desc.host_ptr(),
                             strm_desc.memory_size(), cudaMemcpyHostToDevice));
    CUDA_TRY(CompressOrcDataStreams(
        compressed_data.data(), strm_desc.device_ptr(), chunks.device_ptr(),
        comp_in.device_ptr(), comp_out.device_ptr(), num_stripe_streams,
        num_compressed_blocks, compression_kind_, compression_blocksize_));
    CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(), strm_desc.device_ptr(),
                             strm_desc.memory_size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpyAsync(comp_out.host_ptr(), comp_out.device_ptr(),
                             comp_out.memory_size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));
  }

  ProtobufWriter pbw_(&buffer_);

  // Write file header
  outfile_.write(MAGIC, std::strlen(MAGIC));

  // Write stripes
  size_t group = 0;
  for (size_t stripe_id = 0; stripe_id < stripes.size(); stripe_id++) {
    auto groups_in_stripe = div_by_rowgroups(stripes[stripe_id].numberOfRows);
    stripes[stripe_id].offset = outfile_.tellp();

    // Column (skippable) index streams appear at the start of the stripe
    stripes[stripe_id].indexLength = 0;
    for (size_t col_id = 0; col_id <= (size_t)num_columns; col_id++) {
      write_index_stream(stripe_id, col_id, orc_columns.data(), num_columns,
                         num_data_streams, group, groups_in_stripe, chunks,
                         strm_desc, comp_out, stripes[stripe_id], streams,
                         pbw_);
    }

    // Column data consisting one or more separate streams
    stripes[stripe_id].dataLength = 0;
    for (size_t i = 0; i < num_data_streams; i++) {
      const auto &ss = strm_desc[stripe_id * num_data_streams + i];
      const auto &ck = chunks[group * num_columns + ss.column_id];

      write_data_stream(ss, ck, compressed_data.data(), stream_output.get(),
                        stripes[stripe_id], streams);
    }

    // Write stripefooter consisting of stream information
    StripeFooter sf;
    sf.streams = streams;
    sf.columns.resize(num_columns + 1);
    sf.columns[0].kind = DIRECT;
    sf.columns[0].dictionarySize = 0;
    for (size_t i = 1; i < sf.columns.size(); ++i) {
      sf.columns[i].kind = orc_columns[i - 1].orc_encoding();
      sf.columns[i].dictionarySize =
          (sf.columns[i].kind == DICTIONARY_V2)
              ? orc_columns[i - 1].host_stripe_dict(stripe_id)->num_strings
              : 0;
      if (orc_columns[i - 1].orc_kind() == TIMESTAMP) {
        sf.writerTimezone = "UTC";
      }
    }
    buffer_.resize((compression_kind_ != NONE) ? 3 : 0);
    pbw_.write(&sf);
    stripes[stripe_id].footerLength = buffer_.size();
    if (compression_kind_ != NONE) {
      uint32_t uncomp_sf_len = (stripes[stripe_id].footerLength - 3) * 2 + 1;
      buffer_[0] = static_cast<uint8_t>(uncomp_sf_len >> 0);
      buffer_[1] = static_cast<uint8_t>(uncomp_sf_len >> 8);
      buffer_[2] = static_cast<uint8_t>(uncomp_sf_len >> 16);
    }
    outfile_.write(reinterpret_cast<char *>(buffer_.data()), buffer_.size());

    group += groups_in_stripe;
  }

  // Write filefooter metadata
  // TBD: We may want to add pandas or spark column metadata strings here
  FileFooter ff;
  ff.headerLength = std::strlen(MAGIC);
  ff.contentLength = outfile_.tellp();
  ff.stripes = std::move(stripes);
  ff.numberOfRows = num_rows;
  ff.rowIndexStride = row_index_stride_;
  ff.types.resize(1 + num_columns);
  ff.types[0].kind = STRUCT;
  ff.types[0].subtypes.resize(num_columns);
  ff.types[0].fieldNames.resize(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    ff.types[1 + i].kind = orc_columns[i].orc_kind();
    ff.types[0].subtypes[i] = 1 + i;
    ff.types[0].fieldNames[i] = orc_columns[i].orc_name();
  }
  buffer_.resize((compression_kind_ != NONE) ? 3 : 0);
  pbw_.write(&ff);

  // Write postscript metadata
  PostScript ps;
  ps.footerLength = buffer_.size();
  ps.compression = compression_kind_;
  ps.compressionBlockSize = compression_blocksize_;
  ps.version = {0, 12};
  ps.metadataLength = 0;  // TODO: Write stripe statistics
  ps.magic = MAGIC;
  if (compression_kind_ != NONE) {
    // TODO: If the file footer ends up larger than the compression block
    // size, we'll need to insert additional 3-byte block headers
    uint32_t uncomp_ff_len = (uint32_t)(ps.footerLength - 3) * 2 + 1;
    buffer_[0] = static_cast<uint8_t>(uncomp_ff_len >> 0);
    buffer_[1] = static_cast<uint8_t>(uncomp_ff_len >> 8);
    buffer_[2] = static_cast<uint8_t>(uncomp_ff_len >> 16);
  }
  const auto ps_length = static_cast<uint8_t>(pbw_.write(&ps));
  buffer_.push_back(ps_length);
  outfile_.write(reinterpret_cast<char *>(buffer_.data()), buffer_.size());
  outfile_.flush();
}

void writer::Impl::init_dictionaries(
    orc_column *columns, size_t num_rows, const std::vector<int> &str_col_ids,
    const device_buffer<uint32_t> &dict_data,
    const device_buffer<uint32_t> &dict_index,
    hostdevice_vector<gpu::DictionaryChunk> &dict) {
  const size_t num_rowgroups = dict.size() / str_col_ids.size();

  // Setup per-rowgroup dictionary indexes for each dictionary-aware column
  for (size_t i = 0; i < str_col_ids.size(); ++i) {
    columns[str_col_ids[i]].set_dict_stride(str_col_ids.size());
    columns[str_col_ids[i]].attach_dict_chunk(dict.host_ptr(),
                                              dict.device_ptr());

    for (size_t g = 0; g < num_rowgroups; g++) {
      auto *ck = &dict[g * str_col_ids.size() + str_col_ids[i]];
      ck->valid_map_base = columns[i].orc_valid();
      ck->column_data_base = columns[i].orc_data();
      ck->dict_data = dict_data.data() + i * num_rows + g * row_index_stride_;
      ck->dict_index = dict_index.data() + i * num_rows;  // Indexed by abs row
      ck->start_row = g * row_index_stride_;
      ck->num_rows = std::min<uint32_t>(
          row_index_stride_,
          std::max<int>(columns[i]->size - ck->start_row, 0));
      ck->num_strings = 0;
      ck->string_char_count = 0;
      ck->num_dict_strings = 0;
      ck->dict_char_count = 0;
    }
  }

  CUDA_TRY(cudaMemcpyAsync(dict.device_ptr(), dict.host_ptr(),
                           dict.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(InitDictionaryIndices(dict.device_ptr(), str_col_ids.size(),
                                 num_rowgroups));
  CUDA_TRY(cudaMemcpyAsync(dict.host_ptr(), dict.device_ptr(),
                           dict.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));
}

void writer::Impl::build_dictionaries(
    orc_column *columns, size_t num_rows, const std::vector<int> &str_col_ids,
    const std::vector<uint32_t> &stripe_list,
    const hostdevice_vector<gpu::DictionaryChunk> &dict,
    const device_buffer<uint32_t> &dict_index,
    hostdevice_vector<gpu::StripeDictionary> &stripe_dict) {
  const auto num_rowgroups = dict.size() / str_col_ids.size();

  for (size_t i = 0; i < str_col_ids.size(); i++) {
    size_t direct_cost = 0, dict_cost = 0;
    columns[str_col_ids[i]].attach_stripe_dict(stripe_dict.host_ptr(),
                                               stripe_dict.device_ptr());

    for (size_t j = 0, g = 0; j < stripe_list.size(); j++) {
      const auto num_chunks = stripe_list[j];
      auto *sd = &stripe_dict[j * str_col_ids.size() + i];
      sd->column_data_base = columns[i].host_dict_chunk(0)->column_data_base;
      sd->dict_data = columns[i].host_dict_chunk(g)->dict_data;
      sd->dict_index = dict_index.data() + i * num_rows;  // Indexed by abs row
      sd->column_id = str_col_ids[i];
      sd->start_chunk = (uint32_t)g;
      sd->num_chunks = num_chunks;
      sd->num_strings = 0;
      sd->dict_char_count = 0;
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

  CUDA_TRY(cudaMemcpyAsync(stripe_dict.device_ptr(), stripe_dict.host_ptr(),
                           stripe_dict.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(BuildStripeDictionaries(
      stripe_dict.device_ptr(), stripe_dict.host_ptr(), dict.device_ptr(),
      stripe_list.size(), num_rowgroups, str_col_ids.size()));
  CUDA_TRY(cudaMemcpyAsync(stripe_dict.host_ptr(), stripe_dict.device_ptr(),
                           stripe_dict.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));
}

std::vector<Stream> writer::Impl::gather_streams(
    orc_column *columns, size_t num_columns, size_t num_rows,
    const std::vector<uint32_t> &stripe_list, std::vector<int32_t> &strm_ids) {
  // First n + 1 streams are row index streams, including 'column 0'
  std::vector<Stream> streams;
  streams.resize(num_columns + 1);
  streams[0].column = 0;
  streams[0].kind = ROW_INDEX;
  streams[0].length = 0;

  for (size_t i = 0; i < num_columns; ++i) {
    TypeKind kind = to_orc_type(columns[i]->dtype);
    StreamKind data_kind = DATA;
    StreamKind data2_kind = LENGTH;
    ColumnEncodingKind encoding_kind = DIRECT;

    int64_t present_stream_size = 0;
    int64_t data_stream_size = 0;
    int64_t data2_stream_size = 0;
    int64_t dict_stream_size = 0;
    if (columns[i]->null_count != 0 ||
        columns[i]->size != static_cast<gdf_size_type>(num_rows)) {
      present_stream_size = ((row_index_stride_ + 7) >> 3);
      present_stream_size += (present_stream_size + 0x7f) >> 7;
    }

    switch (kind) {
      case BOOLEAN:
        data_stream_size = div_rowgroups_by<int64_t>(1024) * (128 + 1);
        encoding_kind = DIRECT;
        break;
      case BYTE:
        data_stream_size = div_rowgroups_by<int64_t>(128) * (128 + 1);
        encoding_kind = DIRECT;
        break;
      case SHORT:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 2 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case FLOAT:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size = (columns[i]->null_count != 0)
                               ? div_rowgroups_by<int64_t>(512) * (512 * 4 + 2)
                               : INT64_C(-1);
        encoding_kind = DIRECT;
        break;
      case INT:
      case DATE:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case DOUBLE:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size = (columns[i]->null_count != 0)
                               ? div_rowgroups_by<int64_t>(512) * (512 * 8 + 2)
                               : INT64_C(-1);
        encoding_kind = DIRECT;
        break;
      case LONG:
        data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 8 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case STRING: {
        bool enable_dict = enable_dictionary_;
        size_t direct_data_size = 0;
        size_t dict_data_size = 0;
        size_t dict_strings = 0;
        size_t dict_lengths_div512 = 0;
        for (size_t stripe = 0, g = 0; stripe < stripe_list.size(); stripe++) {
          const auto sd = columns[i].host_stripe_dict(stripe);
          enable_dict = (enable_dict && sd->dict_data != nullptr);
          if (enable_dict) {
            dict_strings += sd->num_strings;
            dict_lengths_div512 += (sd->num_strings + 0x1ff) >> 9;
            dict_data_size += sd->dict_char_count;
          }

          for (uint32_t k = 0; k < stripe_list[stripe]; k++, g++) {
            direct_data_size +=
                columns[i].host_dict_chunk(g)->string_char_count;
          }
        }
        if (enable_dict) {
          uint32_t dict_bits = 0;
          for (dict_bits = 1; dict_bits < 32; dict_bits <<= 1) {
            if (dict_strings <= (1ull << dict_bits)) break;
          }
          const auto valid_count = columns[i]->size - columns[i]->null_count;
          dict_data_size += (dict_bits * valid_count + 7) >> 3;
        }

        // Decide between direct or dictionary encoding
        if (enable_dict && dict_data_size < direct_data_size) {
          data_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
          data2_stream_size = dict_lengths_div512 * (512 * 4 + 2);
          dict_stream_size = std::max<size_t>(dict_data_size, 1);
          encoding_kind = DICTIONARY_V2;
        } else {
          data_stream_size = std::max<size_t>(direct_data_size, 1);
          data2_stream_size = div_rowgroups_by<int64_t>(512) * (512 * 4 + 2);
          encoding_kind = DIRECT_V2;
        }
        break;
      }
      case TIMESTAMP:
        data_stream_size = ((row_index_stride_ + 0x1ff) >> 9) * (512 * 4 + 2);
        data2_stream_size = data_stream_size;
        data2_kind = SECONDARY;
        encoding_kind = DIRECT_V2;
        break;
      default:
        CUDF_FAIL("Unsupported ORC type kind");
    }

    // Initialize the column's metadata
    columns[i].set_orc_kind(kind);
    columns[i].set_orc_encoding(encoding_kind);

    // Initialize the column's index stream
    const auto id = static_cast<uint32_t>(1 + i);
    streams[id].column = id;
    streams[id].kind = ROW_INDEX;
    streams[id].length = 0;

    // Initialize the column's data stream(s)
    const auto base = i * gpu::CI_NUM_STREAMS;
    if (present_stream_size != 0) {
      auto len = static_cast<uint64_t>(present_stream_size);
      strm_ids[base + gpu::CI_PRESENT] = streams.size();
      streams.push_back(orc::Stream{PRESENT, id, len});
    }
    if (data_stream_size != 0) {
      auto len = static_cast<uint64_t>(std::max<int64_t>(data_stream_size, 0));
      strm_ids[base + gpu::CI_DATA] = streams.size();
      streams.push_back(orc::Stream{data_kind, id, len});
    }
    if (data2_stream_size != 0) {
      auto len = static_cast<uint64_t>(std::max<int64_t>(data2_stream_size, 0));
      strm_ids[base + gpu::CI_DATA2] = streams.size();
      streams.push_back(orc::Stream{data2_kind, id, len});
    }
    if (dict_stream_size != 0) {
      auto len = static_cast<uint64_t>(dict_stream_size);
      strm_ids[base + gpu::CI_DICTIONARY] = streams.size();
      streams.push_back(orc::Stream{DICTIONARY_DATA, id, len});
    }
  }

  return streams;
}

device_buffer<uint8_t> writer::Impl::encode_columns(
    orc_column *columns, size_t num_columns, size_t num_rows,
    size_t num_rowgroups, const std::vector<int> &str_col_ids,
    const std::vector<uint32_t> &stripe_list,
    const std::vector<Stream> &streams, const std::vector<int32_t> &strm_ids,
    hostdevice_vector<gpu::EncChunk> &chunks) {
  // Allocate combined buffer for RLE data and string data output
  std::vector<size_t> strm_offsets(streams.size());
  size_t str_data_size = 0;
  auto output = [&]() {
    size_t rle_data_size = 0;
    for (size_t i = 0; i < streams.size(); ++i) {
      const auto &stream = streams[i];
      const auto &column = columns[stream.column - 1];

      if (((stream.kind == DICTIONARY_DATA || stream.kind == LENGTH) &&
           (column.orc_encoding() == DICTIONARY_V2)) ||
          ((stream.kind == DATA) && (column.orc_kind() == STRING &&
                                     column.orc_encoding() == DIRECT_V2))) {
        strm_offsets[i] = str_data_size;
        str_data_size += stream.length;
      } else {
        strm_offsets[i] = rle_data_size;
        rle_data_size += (stream.length * num_rowgroups + 7) & ~7;
      }
    }
    str_data_size = (str_data_size + 7) & ~7;

    return device_buffer<uint8_t>(rle_data_size + str_data_size);
  }();

  // Initialize column chunks' descriptions
  size_t stripe_start = 0;
  size_t stripe_id = 0;
  for (size_t j = 0; j < num_rowgroups; j++) {
    for (size_t i = 0; i < num_columns; i++) {
      auto *ck = &chunks[j * num_columns + i];
      ck->start_row = (j * row_index_stride_);
      ck->num_rows =
          std::min<uint32_t>(row_index_stride_, num_rows - ck->start_row);
      ck->valid_rows = columns[i]->size;
      ck->encoding_kind = columns[i].orc_encoding();
      ck->type_kind = columns[i].orc_kind();
      if (ck->type_kind == STRING) {
        ck->valid_map_base = columns[i].orc_valid();
        ck->column_data_base =
            (ck->encoding_kind == DICTIONARY_V2)
                ? columns[i].host_stripe_dict(stripe_id)->dict_index
                : columns[i].orc_data();
        ck->dtype_len = 1;
      } else {
        ck->valid_map_base = columns[i].orc_valid();
        ck->column_data_base = columns[i].orc_data();
        ck->dtype_len = cudf::size_of(columns[i]->dtype);
      }
      ck->scale = to_clockscale<uint8_t>(columns[i]->dtype_info.time_unit);

      for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
        const auto strm_id = strm_ids[i * gpu::CI_NUM_STREAMS + k];

        ck->strm_id[k] = strm_id;
        if (strm_id >= 0) {
          if ((k == gpu::CI_DICTIONARY) ||
              (k == gpu::CI_DATA2 && ck->encoding_kind == DICTIONARY_V2)) {
            if (j == stripe_start) {
              const int32_t dict_stride = columns[i].get_dict_stride();
              const auto stripe = columns[i].host_stripe_dict(stripe_id);
              ck->strm_len[k] =
                  (k == gpu::CI_DICTIONARY)
                      ? stripe->dict_char_count
                      : (((stripe->num_strings + 0x1ff) >> 9) * (512 * 4 + 2));
              if (stripe_id == 0) {
                ck->streams[k] = output.data() + strm_offsets[strm_id];
              } else {
                const auto *ck_up =
                    &chunks[stripe[-dict_stride].start_chunk * num_columns + i];
                ck->streams[k] = ck_up->streams[k] + ck_up->strm_len[k];
              }
            } else {
              ck->strm_len[k] = 0;
              ck->streams[k] = ck[-num_columns].streams[k];
            }
          } else if (k == gpu::CI_DATA && ck->type_kind == STRING &&
                     ck->encoding_kind == DIRECT_V2) {
            ck->strm_len[k] = columns[i].host_dict_chunk(j)->string_char_count;
            ck->streams[k] = (j == 0) ? output.data() + strm_offsets[strm_id]
                                      : (ck[-num_columns].streams[k] +
                                         ck[-num_columns].strm_len[k]);
          } else if (k == gpu::CI_DATA && streams[strm_id].length == 0 &&
                     (ck->type_kind == DOUBLE || ck->type_kind == FLOAT)) {
            // Pass-through
            ck->strm_len[k] = ck->num_rows * ck->dtype_len;
            ck->streams[k] = nullptr;
          } else {
            ck->strm_len[k] = streams[strm_id].length;
            ck->streams[k] = output.data() + str_data_size +
                             strm_offsets[strm_id] +
                             streams[strm_id].length * j;
          }
        } else {
          ck->strm_len[k] = 0;
          ck->streams[k] = nullptr;
        }
      }
    }

    // Track the current stripe this rowgroup chunk belongs
    if (j + 1 == stripe_start + stripe_list[stripe_id]) {
      stripe_start = j + 1;
      stripe_id++;
    }
  }

  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice));
  if (!str_col_ids.empty()) {
    auto d_stripe_dict = columns[str_col_ids[0]].device_stripe_dict();
    CUDA_TRY(EncodeStripeDictionaries(d_stripe_dict, chunks.device_ptr(),
                                      str_col_ids.size(), num_columns,
                                      stripe_list.size()));
  }
  CUDA_TRY(
      EncodeOrcColumnData(chunks.device_ptr(), num_columns, num_rowgroups));
  CUDA_TRY(cudaStreamSynchronize(0));

  return output;
}

std::vector<StripeInformation> writer::Impl::gather_stripes(
    size_t num_columns, size_t num_rows, size_t num_index_streams,
    size_t num_data_streams, const std::vector<uint32_t> &stripe_list,
    hostdevice_vector<gpu::EncChunk> &chunks,
    hostdevice_vector<gpu::StripeStream> &strm_desc) {
  std::vector<StripeInformation> stripes(stripe_list.size());
  size_t group = 0;
  size_t stripe_start = 0;
  for (size_t s = 0; s < stripe_list.size(); s++) {
    size_t stripe_group_end = group + stripe_list[s];

    for (size_t i = 0; i < num_columns; i++) {
      const auto *ck = &chunks[group * num_columns + i];

      // Assign stream data of column data stream(s)
      for (int k = 0; k < gpu::CI_INDEX; k++) {
        const auto stream_id = ck->strm_id[k];
        if (stream_id != -1) {
          auto *ss =
              &strm_desc[s * num_data_streams + stream_id - num_index_streams];
          ss->stream_size = 0;
          ss->first_chunk_id = (group * num_columns + i);
          ss->num_chunks = (stripe_group_end - group);
          ss->column_id = i;
          ss->strm_type = k;
        }
      }
    }

    group = stripe_group_end;
    size_t stripe_end = std::min(group * row_index_stride_, num_rows);
    stripes[s].numberOfRows = stripe_end - stripe_start;
    stripe_start = stripe_end;
  }

  CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(), strm_desc.host_ptr(),
                           strm_desc.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(CompactOrcDataStreams(strm_desc.device_ptr(), chunks.device_ptr(),
                                 strm_desc.size(), num_columns));
  CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(), strm_desc.device_ptr(),
                           strm_desc.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  return stripes;
}

void writer::Impl::write_index_stream(
    int32_t stripe_id, int32_t stream_id, orc_column *columns,
    size_t num_columns, size_t num_data_streams, size_t group,
    size_t groups_in_stripe, const hostdevice_vector<gpu::EncChunk> &chunks,
    const hostdevice_vector<gpu::StripeStream> &strm_desc,
    const hostdevice_vector<gpu_inflate_status_s> &comp_out,
    StripeInformation &stripe, std::vector<Stream> &streams,
    ProtobufWriter &pbw_) {
  // 0: position, 1: block position, 2: compressed position, 3: compressed size
  std::array<int32_t, 4> present;
  std::array<int32_t, 4> data;
  std::array<int32_t, 4> data2;
  auto kind = STRUCT;

  auto find_record = [=, &strm_desc](const gpu::EncChunk &chunk,
                                     const gpu::StreamIndexType type) {
    std::array<int32_t, 4> record{-1, -1, -1, -1};
    if (chunk.strm_id[type] > 0) {
      record[0] = 0;
      if (compression_kind_ != NONE) {
        const auto *ss = &strm_desc[stripe_id * num_data_streams +
                                    chunk.strm_id[type] - (num_columns + 1)];
        record[1] = ss->first_block;
        record[2] = 0;
        record[3] = ss->stream_size;
      }
    }
    return record;
  };
  auto scan_record = [=, &comp_out](const gpu::EncChunk &chunk,
                                    const gpu::StreamIndexType type,
                                    std::array<int32_t, 4> &record) {
    if (record[0] >= 0) {
      record[0] += chunk.strm_len[type];
      while ((record[1] >= 0) &&
             (static_cast<size_t>(record[0]) >= compression_blocksize_) &&
             (record[3] + 3 + comp_out[record[1]].bytes_written <
              static_cast<size_t>(record[4]))) {
        record[0] -= compression_blocksize_;
        record[3] += 3 + comp_out[record[1]].bytes_written;
        record[1] += 1;
      }
    }
  };

  // TBD: Not sure we need an empty index stream for column 0
  if (stream_id != 0) {
    const auto &ck = chunks[stream_id - 1];
    present = find_record(ck, gpu::CI_PRESENT);
    data = find_record(ck, gpu::CI_DATA);
    data2 = find_record(ck, gpu::CI_DATA2);

    // Change string dictionary to int from index point of view
    kind = columns[stream_id - 1].orc_kind();
    if (kind == STRING &&
        columns[stream_id - 1].orc_encoding() == DICTIONARY_V2) {
      kind = INT;
    }
  }

  buffer_.resize((compression_kind_ != NONE) ? 3 : 0);

  // Add row index entries
  for (size_t g = group; g < group + groups_in_stripe; g++) {
    pbw_.put_row_index_entry(present[2], present[0], data[2], data[0], data2[2],
                             data2[0], kind);

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
    buffer_[0] = static_cast<uint8_t>(uncomp_ix_len >> 0);
    buffer_[1] = static_cast<uint8_t>(uncomp_ix_len >> 8);
    buffer_[2] = static_cast<uint8_t>(uncomp_ix_len >> 16);
  }
  outfile_.write(reinterpret_cast<char *>(buffer_.data()), buffer_.size());
  stripe.indexLength += buffer_.size();
}

void writer::Impl::write_data_stream(const gpu::StripeStream &strm_desc,
                                     const gpu::EncChunk &chunk,
                                     const uint8_t *compressed_data,
                                     uint8_t *stream_out,
                                     StripeInformation &stripe,
                                     std::vector<Stream> &streams) {
  const auto length = strm_desc.stream_size;
  streams[chunk.strm_id[strm_desc.strm_type]].length = length;
  if (length != 0) {
    const auto *stream_in = (compression_kind_ == NONE)
                                ? chunk.streams[strm_desc.strm_type]
                                : (compressed_data + strm_desc.bfr_offset);
    CUDA_TRY(cudaMemcpyAsync(stream_out, stream_in, length,
                             cudaMemcpyDeviceToHost, 0));
    CUDA_TRY(cudaStreamSynchronize(0));

    outfile_.write(reinterpret_cast<char *>(stream_out), length);
  }
  stripe.dataLength += length;
}

writer::writer(std::string filepath, writer_options const &options)
    : impl_(std::make_unique<Impl>(filepath, options)) {}

void writer::write_all(const cudf::table &table) { impl_->write(table); }

writer::~writer() = default;

}  // namespace orc
}  // namespace io
}  // namespace cudf
