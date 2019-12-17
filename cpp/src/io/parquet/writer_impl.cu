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

/**
 * @file writer_impl.cu
 * @brief cuDF-IO parquet writer class implementation
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
namespace experimental {
namespace io {
namespace detail {
namespace parquet {

using namespace cudf::io::parquet;
using namespace cudf::io;

namespace {

/**
 * @brief Helper for pinned host memory
 **/
template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

/**
 * @brief Function that translates GDF compression to parquet compression
 **/
parquet::Compression to_parquet_compression(
    compression_type compression) {
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY:
      return parquet::Compression::SNAPPY;
    case compression_type::NONE:
      return parquet::Compression::UNCOMPRESSED;
    default:
      CUDF_EXPECTS(false, "Unsupported compression type");
      return parquet::Compression::UNCOMPRESSED;
  }
}

}  // namespace


/**
 * @brief Helper kernel for converting string data/offsets into nvstrdesc
 * REMOVEME: Once we eliminate the legacy readers/writers, the kernels could be
 * made to use the native offset+data layout.
 **/
__global__ void stringdata_to_nvstrdesc(gpu::nvstrdesc_s *dst, const size_type *offsets,
                        const char *strdata, const uint32_t *nulls,
                        size_type column_size) {
  size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < column_size) {
    uint32_t is_valid = (nulls) ? (nulls[row >> 5] >> (row & 0x1f)) & 1 : 1;
    size_t count;
    const char *ptr;
    if (is_valid) {
      size_type cur = offsets[row];
      size_type next = offsets[row + 1];
      ptr = strdata + cur;
      count = (next > cur) ? next - cur : 0;
    }
    else {
      ptr = nullptr;
      count = 0;
    }
    dst[row].ptr = ptr;
    dst[row].count = count;
  }
}


/**
 * @brief Helper class that adds parquet-specific column info
 **/
class parquet_column_view {
 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   **/
  explicit parquet_column_view(size_t id, column_view const &col, const table_metadata *metadata, cudaStream_t stream)
      : _id(id),
        _string_type(col.type().id() == type_id::STRING),
        _type_width(_string_type ? 0 : cudf::size_of(col.type())),
        _converted_type(ConvertedType::UNKNOWN),
        _ts_scale(0),
        _data_count(col.size()),
        _null_count(col.null_count()),
        _data(col.data<uint8_t>()),
        _nulls(col.has_nulls() ? col.null_mask() : nullptr) {
    switch(col.type().id()) {
      case cudf::type_id::INT8:
        _physical_type = Type::INT32;
        _converted_type = ConvertedType::INT_8;
        _stats_dtype = statistics_dtype::dtype_int8;
        break;
      case cudf::type_id::INT16:
        _physical_type = Type::INT32;
        _converted_type = ConvertedType::INT_16;
        _stats_dtype = statistics_dtype::dtype_int16;
        break;
      case cudf::type_id::INT32:
      case cudf::type_id::CATEGORY:
        _physical_type = Type::INT32;
        _stats_dtype = statistics_dtype::dtype_int32;
        break;
      case cudf::type_id::INT64:
        _physical_type = Type::INT64;
        _stats_dtype = statistics_dtype::dtype_int64;
        break;
      case cudf::type_id::FLOAT32:
        _physical_type = Type::FLOAT;
        _stats_dtype = statistics_dtype::dtype_float32;
        break;
      case cudf::type_id::FLOAT64:
        _physical_type = Type::DOUBLE;
        _stats_dtype = statistics_dtype::dtype_float64;
        break;
      case cudf::type_id::BOOL8:
        _physical_type = Type::BOOLEAN;
        _stats_dtype = statistics_dtype::dtype_bool8;
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        _physical_type = Type::INT32;
        _converted_type = ConvertedType::DATE;
        _stats_dtype = statistics_dtype::dtype_int32;
        break;
      case cudf::type_id::TIMESTAMP_SECONDS:
        _physical_type = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MILLIS;
        _stats_dtype = statistics_dtype::dtype_timestamp64;
        _ts_scale = 1000;
        break;
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        _physical_type = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MILLIS;
        _stats_dtype = statistics_dtype::dtype_timestamp64;
        break;
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        _physical_type = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MICROS;
        _stats_dtype = statistics_dtype::dtype_timestamp64;
        break;
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        _physical_type = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MICROS;
        _stats_dtype = statistics_dtype::dtype_timestamp64;
        _ts_scale = -1000;
        break;
    case cudf::type_id::STRING:
        _physical_type = Type::BYTE_ARRAY;
        //_converted_type = ConvertedType::UTF8; // TBD
        _stats_dtype = statistics_dtype::dtype_string;
        break;
    default:
        _physical_type = UNDEFINED_TYPE;
        _stats_dtype = dtype_none;
        break;
    }
    if (_string_type && _data_count > 0) {
      strings_column_view view{col};
      _indexes = rmm::device_buffer(_data_count * sizeof(gpu::nvstrdesc_s), stream);
      stringdata_to_nvstrdesc<<< ((_data_count-1)>>8)+1, 256, 0, stream >>>(
            reinterpret_cast<gpu::nvstrdesc_s *>(_indexes.data()),
            view.offsets().data<size_type>(), view.chars().data<char>(),
            _nulls, _data_count);
      _data = _indexes.data();
      cudaStreamSynchronize(stream);
    }
    // Generating default name if name isn't present in metadata
    if (metadata && _id < metadata->column_names.size()) {
      _name = metadata->column_names[_id];
    }
    else {
      _name = "_col" + std::to_string(_id);
    }
  }

  auto is_string() const noexcept { return _string_type; }
  size_t type_width() const noexcept { return _type_width; }
  size_t data_count() const noexcept { return _data_count; }
  size_t null_count() const noexcept { return _null_count; }
  void const *data() const noexcept { return _data; }
  uint32_t const *nulls() const noexcept { return _nulls; }

  auto name() const noexcept { return _name; }
  auto physical_type() const noexcept { return _physical_type; }
  auto converted_type() const noexcept { return _converted_type; }
  auto stats_type() const noexcept { return _stats_dtype; }
  int32_t ts_scale() const noexcept { return _ts_scale; }

  // Dictionary management
  uint32_t *get_dict_data() { return (_dict_data.size()) ? _dict_data.data().get() : nullptr; }
  uint32_t *get_dict_index() { return (_dict_index.size()) ? _dict_index.data().get() : nullptr; }
  void use_dictionary(bool use_dict) { _dictionary_used = use_dict; }
  void alloc_dictionary(size_t max_num_rows) {
    _dict_data.resize(max_num_rows);
    _dict_index.resize(max_num_rows);
  }
  bool check_dictionary_used() {
    if (!_dictionary_used) {
      _dict_data.resize(0);
      _dict_data.shrink_to_fit();
      _dict_index.resize(0);
      _dict_index.shrink_to_fit();
    }
    return _dictionary_used;
  }

 private:
  // Identifier within set of columns
  size_t _id = 0;
  bool _string_type = false;

  size_t _type_width = 0;
  size_t _data_count = 0;
  size_t _null_count = 0;
  void const *_data = nullptr;
  uint32_t const *_nulls = nullptr;

  // parquet-related members
  std::string _name{};
  Type _physical_type;
  ConvertedType _converted_type;
  statistics_dtype _stats_dtype;
  int32_t _ts_scale;

  // Dictionary-related members
  bool _dictionary_used = false;
  rmm::device_vector<uint32_t> _dict_data;
  rmm::device_vector<uint32_t> _dict_index;

  // String-related members
  rmm::device_buffer _indexes;
};


void writer::impl::init_page_fragments(hostdevice_vector<gpu::PageFragment>& frag,
                                       hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                       uint32_t num_columns, uint32_t num_fragments,
                                       uint32_t num_rows, uint32_t fragment_size,
                                       cudaStream_t stream) {

  CUDA_TRY(cudaMemcpyAsync(col_desc.device_ptr(), col_desc.host_ptr(),
                           col_desc.memory_size(), cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::InitPageFragments(frag.device_ptr(), col_desc.device_ptr(),
                                  num_fragments, num_columns, fragment_size,
                                  num_rows, stream));
  CUDA_TRY(cudaMemcpyAsync(frag.host_ptr(), frag.device_ptr(),
                           frag.memory_size(), cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}


void writer::impl::gather_fragment_statistics(statistics_chunk *frag_stats_chunk,
                                              hostdevice_vector<gpu::PageFragment>& frag,
                                              hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                              uint32_t num_columns, uint32_t num_fragments,
                                              uint32_t fragment_size, cudaStream_t stream) {
  rmm::device_vector<statistics_group> frag_stats_group(num_fragments * num_columns);

  CUDA_TRY(gpu::InitFragmentStatistics(frag_stats_group.data().get(), frag.device_ptr(),
                                       col_desc.device_ptr(), num_fragments, num_columns,
                                       fragment_size, stream));
  CUDA_TRY(GatherColumnStatistics(frag_stats_chunk, frag_stats_group.data().get(),
                                  num_fragments * num_columns, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}


void writer::impl::build_chunk_dictionaries(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                                            hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                            uint32_t num_rowgroups, uint32_t num_columns,
                                            uint32_t num_dictionaries, cudaStream_t stream) {
  size_t dict_scratch_size = (size_t)num_dictionaries * gpu::kDictScratchSize;
  rmm::device_vector<uint32_t> dict_scratch(dict_scratch_size / sizeof(uint32_t));
  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::BuildChunkDictionaries(chunks.device_ptr(), dict_scratch.data().get(),
                                  dict_scratch_size, num_rowgroups * num_columns, stream));
  CUDA_TRY(gpu::InitEncoderPages(chunks.device_ptr(), nullptr, col_desc.device_ptr(),
                            num_rowgroups, num_columns, nullptr, nullptr, stream));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}


void writer::impl::init_encoder_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                                      hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                      gpu::EncPage *pages,
                                      statistics_chunk *page_stats,
                                      statistics_chunk *frag_stats,
                                      uint32_t num_rowgroups, uint32_t num_columns,
                                      uint32_t num_pages, uint32_t num_stats_bfr,
                                     cudaStream_t stream) {
  rmm::device_vector<statistics_merge_group> page_stats_mrg(num_stats_bfr);
  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(InitEncoderPages(chunks.device_ptr(), pages, col_desc.device_ptr(),
                            num_rowgroups, num_columns,
                            (num_stats_bfr) ? page_stats_mrg.data().get() : nullptr,
                            (num_stats_bfr > num_pages) ? page_stats_mrg.data().get() + num_pages : nullptr,
                            stream));
  if (num_stats_bfr > 0) {
    CUDA_TRY(MergeColumnStatistics(page_stats, frag_stats,
                                   page_stats_mrg.data().get(), num_pages, stream));
    if (num_stats_bfr > num_pages) {
      CUDA_TRY(MergeColumnStatistics(page_stats + num_pages, page_stats,
                                   page_stats_mrg.data().get() + num_pages, num_stats_bfr - num_pages, stream));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));
}


void writer::impl::encode_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                                gpu::EncPage *pages, uint32_t num_columns,
                                uint32_t pages_in_batch, uint32_t first_page_in_batch,
                                uint32_t rowgroups_in_batch, uint32_t first_rowgroup,
                                gpu_inflate_input_s *comp_in,
                                gpu_inflate_status_s *comp_out,
                                const statistics_chunk *page_stats,
                                const statistics_chunk *chunk_stats,
                                cudaStream_t stream) {
  CUDA_TRY(gpu::EncodePages(pages, chunks.device_ptr(), pages_in_batch,
                            first_page_in_batch, comp_in, comp_out, stream));
  switch(compression_) {
   case parquet::Compression::SNAPPY:
    CUDA_TRY(gpu_snap(comp_in, comp_out, pages_in_batch, stream));
    break;
   default:
    break;
  }
  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the chunk-level
  CUDA_TRY(DecideCompression(chunks.device_ptr() + first_rowgroup * num_columns,
                             pages, rowgroups_in_batch * num_columns, first_page_in_batch, comp_out, stream));
  CUDA_TRY(EncodePageHeaders(pages, chunks.device_ptr(), pages_in_batch, first_page_in_batch, comp_out,
                             page_stats, chunk_stats, stream));
  CUDA_TRY(GatherPages(chunks.device_ptr() + first_rowgroup * num_columns, pages,
                       rowgroups_in_batch * num_columns, stream));
  CUDA_TRY(cudaMemcpyAsync(&chunks[first_rowgroup * num_columns],
                           chunks.device_ptr() + first_rowgroup * num_columns,
                           rowgroups_in_batch * num_columns * sizeof(gpu::EncColumnChunk),
                           cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}


writer::impl::impl(std::string filepath, writer_options const &options,
                   rmm::mr::device_memory_resource *mr)
    : _mr(mr) {
  compression_ = to_parquet_compression(options.compression);
  stats_granularity_ = options.stats_granularity;

  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::impl::write(table_view const &table, const table_metadata *metadata, cudaStream_t stream) {
  size_type num_columns = table.num_columns();
  size_type num_rows = 0;

  // Wrapper around cudf columns to attach parquet-specific type info
  std::vector<parquet_column_view> parquet_columns;
  parquet_columns.reserve(num_columns); // Avoids unnecessary re-allocation
  for (auto it = table.begin(); it < table.end(); ++it) {
    const auto col = *it;
    const auto current_id = parquet_columns.size();

    num_rows = std::max<uint32_t>(num_rows, col.size());
    parquet_columns.emplace_back(current_id, col, metadata, stream);
  }

  // Initialize column description
  FileMetaData md;
  hostdevice_vector<gpu::EncColumnDesc> col_desc(num_columns);
  md.version = 1;
  md.num_rows = num_rows;
  md.schema.resize(1 + num_columns);
  md.schema[0].type = UNDEFINED_TYPE;
  md.schema[0].repetition_type = NO_REPETITION_TYPE;
  md.schema[0].name = "schema";
  md.schema[0].num_children = num_columns;
  md.column_order_listsize = (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_columns : 0;
  if (metadata) {
    for (auto it = metadata->user_data.begin(); it != metadata->user_data.end(); it++) {
      md.key_value_metadata.push_back({it->first, it->second});
    }
  }
  for (auto i = 0; i < num_columns; i++) {
    auto& col = parquet_columns[i];
    // Column metadata
    md.schema[1 + i].type = col.physical_type();
    md.schema[1 + i].converted_type = col.converted_type();
    md.schema[1 + i].repetition_type = (col.null_count() || col.data_count() < (size_t)num_rows) ? OPTIONAL : REQUIRED;
    md.schema[1 + i].name = col.name();
    md.schema[1 + i].num_children = 0; // Leaf node
    // GPU column description
    auto *desc = &col_desc[i];
    desc->column_data_base = col.data();
    desc->valid_map_base = col.nulls();
    desc->stats_dtype = col.stats_type();
    desc->ts_scale = col.ts_scale();
    if (md.schema[1 + i].type != BOOLEAN && md.schema[1 + i].type != UNDEFINED_TYPE) {
      col.alloc_dictionary(num_rows);
      desc->dict_index = col.get_dict_index();
      desc->dict_data = col.get_dict_data();
    }
    else {
      desc->dict_data = nullptr;
      desc->dict_index = nullptr;
    }
    desc->num_rows = col.data_count();
    desc->physical_type = static_cast<uint8_t>(md.schema[1 + i].type);
    desc->converted_type = static_cast<uint8_t>(md.schema[1 + i].converted_type);
    desc->level_bits = (md.schema[1 + i].repetition_type == OPTIONAL) ? 1 : 0;
  }

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings will start producing fragments larger than the
  // desired page size -> TODO: keep track of the max fragment size, and iteratively reduce this value if the largest
  // fragment exceeds the max page size limit (we ideally want the page size to be below 1MB so as to have enough pages
  // to get good compression/decompression performance).
  uint32_t fragment_size = 5000;
  uint32_t num_fragments = (uint32_t)((md.num_rows + fragment_size - 1) / fragment_size);
  hostdevice_vector<gpu::PageFragment> fragments(num_columns * num_fragments);
  init_page_fragments(fragments, col_desc, num_columns, num_fragments, num_rows, fragment_size, stream);

  // Decide row group boundaries based on uncompressed data size
  size_t rowgroup_size = 0;
  uint32_t num_rowgroups = 0;
  for (uint32_t f = 0, rowgroup_start = 0; f < num_fragments; f++) {
    size_t fragment_data_size = 0;
    for (auto i = 0; i < num_columns; i++) {
      fragment_data_size += fragments[i * num_fragments + f].fragment_data_size;
    }
    if (f > rowgroup_start && (rowgroup_size + fragment_data_size > max_rowgroup_size_ ||
                               (f + 1 - rowgroup_start) * fragment_size > max_rowgroup_rows_)) {
      md.row_groups.resize(num_rowgroups + 1);
      md.row_groups[num_rowgroups++].num_rows = (f - rowgroup_start) * fragment_size;
      rowgroup_start = f;
      rowgroup_size = 0;
    }
    rowgroup_size += fragment_data_size;
    if (f + 1 == num_fragments) {
      md.row_groups.resize(num_rowgroups + 1);
      md.row_groups[num_rowgroups++].num_rows = md.num_rows - rowgroup_start * fragment_size;
    }
  }

  // Allocate column chunks and gather fragment statistics
  rmm::device_vector<statistics_chunk> frag_stats;
  if (stats_granularity_ != statistics_freq::STATISTICS_NONE) {
    frag_stats.resize(num_fragments * num_columns);
    gather_fragment_statistics(frag_stats.data().get(), fragments, col_desc,
                               num_columns, num_fragments, fragment_size, stream);
  }

  // Initialize row groups and column chunks
  uint32_t num_chunks = num_rowgroups * num_columns;
  hostdevice_vector<gpu::EncColumnChunk> chunks(num_chunks);
  uint32_t num_dictionaries = 0;
  for (uint32_t r = 0, f = 0, start_row = 0; r < num_rowgroups; r++) {
    uint32_t fragments_in_chunk = (uint32_t)((md.row_groups[r].num_rows + fragment_size - 1) / fragment_size);
    md.row_groups[r].total_byte_size = 0;
    md.row_groups[r].columns.resize(num_columns);
    for (int i = 0; i < num_columns; i++) {
      gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
      bool dict_enable = false;

      ck->col_desc = col_desc.device_ptr() + i;
      ck->uncompressed_bfr = nullptr;
      ck->compressed_bfr = nullptr;
      ck->bfr_size = 0;
      ck->compressed_size = 0;
      ck->fragments = fragments.device_ptr() + i * num_fragments + f;
      ck->stats = (frag_stats.size() != 0) ? frag_stats.data().get() + i * num_fragments + f : nullptr;
      ck->start_row = start_row;
      ck->num_rows = (uint32_t)md.row_groups[r].num_rows;
      ck->first_fragment = i * num_fragments + f;
      ck->first_page = 0;
      ck->num_pages = 0;
      ck->is_compressed = 0;
      ck->dictionary_id = num_dictionaries;
      ck->ck_stat_size = 0;
      if (col_desc[i].dict_data) {
        const gpu::PageFragment *ck_frag = &fragments[i * num_fragments + f];
        size_t plain_size = 0;
        size_t dict_size = 1;
        uint32_t num_dict_vals = 0;
        for (uint32_t j = 0; j < fragments_in_chunk && num_dict_vals < 65536; j++)
        {
          plain_size += ck_frag[j].fragment_data_size;
          dict_size += ck_frag[j].dict_data_size + ((num_dict_vals > 256) ? 2 : 1) * ck_frag[j].non_nulls;
          num_dict_vals += ck_frag[j].num_dict_vals;
        }
        if (dict_size < plain_size)
        {
          parquet_columns[i].use_dictionary(true); 
          dict_enable = true;
          num_dictionaries++;
        }
      }
      ck->has_dictionary = dict_enable;
      md.row_groups[r].columns[i].meta_data.type = md.schema[1 + i].type;
      md.row_groups[r].columns[i].meta_data.encodings = { PLAIN, RLE };
      if (dict_enable) {
        md.row_groups[r].columns[i].meta_data.encodings.push_back(PLAIN_DICTIONARY);
      }
      md.row_groups[r].columns[i].meta_data.path_in_schema = { md.schema[1 + i].name };
      md.row_groups[r].columns[i].meta_data.codec = UNCOMPRESSED;
      md.row_groups[r].columns[i].meta_data.num_values = md.row_groups[r].num_rows;
    }
    f += fragments_in_chunk;
    start_row += (uint32_t)md.row_groups[r].num_rows;
  }

  // Free unused dictionaries
  for (auto& col : parquet_columns) {
    col.check_dictionary_used();
  }

  // Build chunk dictionaries and count pages
  build_chunk_dictionaries(chunks, col_desc, num_rowgroups, num_columns, num_dictionaries, stream);

  // Initialize batches of rowgroups to encode (mainly to limit peak memory usage)
  std::vector<uint32_t> batch_list;
  uint32_t num_pages = 0;
  size_t max_bytes_in_batch = 1024 * 1024 * 1024;    // 1GB - TBD: Tune this
  size_t max_uncomp_bfr_size = 0;
  size_t max_chunk_bfr_size = 0;
  uint32_t max_pages_in_batch = 0;
  size_t bytes_in_batch = 0;
  for (uint32_t r = 0, groups_in_batch = 0, pages_in_batch = 0; r <= num_rowgroups; r++) {
    size_t rowgroup_size = 0;
    if (r < num_rowgroups) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        ck->first_page = num_pages;
        num_pages += ck->num_pages;
        pages_in_batch += ck->num_pages;
        rowgroup_size += ck->bfr_size;
        max_chunk_bfr_size = std::max(max_chunk_bfr_size, (size_t)std::max(ck->bfr_size, ck->compressed_size));
      }
    }
    // TBD: We may want to also shorten the batch if we have enough pages (not just based on size)
    if ((r == num_rowgroups) || (groups_in_batch != 0 && bytes_in_batch + rowgroup_size > max_bytes_in_batch)) {
      max_uncomp_bfr_size = std::max(max_uncomp_bfr_size, bytes_in_batch);
      max_pages_in_batch = std::max(max_pages_in_batch, pages_in_batch);
      batch_list.push_back(groups_in_batch);
      groups_in_batch = 0;
      bytes_in_batch = 0;
      pages_in_batch = 0;
    }
    bytes_in_batch += rowgroup_size;
    groups_in_batch++;
  }

  // Initialize data pointers in batch
  size_t max_comp_bfr_size = (compression_ != parquet::Compression::UNCOMPRESSED) ? gpu::GetMaxCompressedBfrSize(max_uncomp_bfr_size, max_pages_in_batch) : 0;
  uint32_t max_comp_pages = (compression_ != parquet::Compression::UNCOMPRESSED) ? max_pages_in_batch : 0;
  uint32_t num_stats_bfr = (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;
  rmm::device_buffer uncomp_bfr(max_uncomp_bfr_size, stream);
  rmm::device_buffer comp_bfr(max_comp_bfr_size, stream);
  rmm::device_vector<gpu_inflate_input_s> comp_in(max_comp_pages);
  rmm::device_vector<gpu_inflate_status_s> comp_out(max_comp_pages);
  rmm::device_vector<gpu::EncPage> pages(num_pages);
  rmm::device_vector<statistics_chunk> page_stats(num_stats_bfr);
  for (uint32_t b = 0, r = 0; b < (uint32_t)batch_list.size(); b++) {
    uint8_t *bfr = reinterpret_cast<uint8_t *>(uncomp_bfr.data());
    uint8_t *bfr_c = reinterpret_cast<uint8_t *>(comp_bfr.data());
    for (uint32_t j = 0; j < batch_list[b]; j++, r++) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        ck->uncompressed_bfr = bfr;
        ck->compressed_bfr = bfr_c;
        bfr += ck->bfr_size;
        bfr_c += ck->compressed_size;
      }
    }
  }
  init_encoder_pages(chunks, col_desc, pages.data().get(),
                     (num_stats_bfr) ? page_stats.data().get() : nullptr,
                     (num_stats_bfr) ? frag_stats.data().get() : nullptr,
                     num_rowgroups, num_columns, num_pages, num_stats_bfr,
                     stream);

  auto host_bfr = [&]() {
    return pinned_buffer<uint8_t>{[](size_t size) {
                                    uint8_t *ptr = nullptr;
                                    CUDA_TRY(cudaMallocHost(&ptr, size));
                                    return ptr;
                                  }(max_chunk_bfr_size),
                                  cudaFreeHost};
  }();

  // Write file header
  file_header_s fhdr;
  fhdr.magic = PARQUET_MAGIC;
  outfile_.write(reinterpret_cast<char *>(&fhdr), sizeof(fhdr));

  // Encode row groups in batches
  size_t current_chunk_offset = sizeof(fhdr);
  for (uint32_t b = 0, r = 0; b < (uint32_t)batch_list.size(); b++) {
    // Count pages in this batch
    uint32_t rnext = r + batch_list[b];
    uint32_t first_page_in_batch = chunks[r * num_columns].first_page;
    uint32_t first_page_in_next_batch = (rnext < num_rowgroups) ? chunks[rnext * num_columns].first_page : num_pages;
    uint32_t pages_in_batch = first_page_in_next_batch - first_page_in_batch;
    encode_pages(chunks, pages.data().get(), num_columns, pages_in_batch, first_page_in_batch,
                 batch_list[b], r, comp_in.data().get(), comp_out.data().get(),
                 (stats_granularity_ == statistics_freq::STATISTICS_PAGE) ? page_stats.data().get() : nullptr,
                 (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? page_stats.data().get() + num_pages : nullptr,
                 stream);
    for (; r < rnext; r++) {
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        void *dev_bfr;
        if (ck->is_compressed) {
          md.row_groups[r].columns[i].meta_data.codec = compression_;
          dev_bfr = ck->compressed_bfr;
        }
        else {
          dev_bfr = ck->uncompressed_bfr;
        }
        CUDA_TRY(cudaMemcpyAsync(host_bfr.get(), dev_bfr, ck->ck_stat_size + ck->compressed_size,
                                 cudaMemcpyDeviceToHost, stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
        md.row_groups[r].total_byte_size += ck->compressed_size;
        md.row_groups[r].columns[i].meta_data.data_page_offset = current_chunk_offset + ((ck->has_dictionary) ? ck->dictionary_size : 0);
        md.row_groups[r].columns[i].meta_data.dictionary_page_offset = (ck->has_dictionary) ? current_chunk_offset : 0;
        md.row_groups[r].columns[i].meta_data.total_uncompressed_size = ck->bfr_size;
        md.row_groups[r].columns[i].meta_data.total_compressed_size = ck->compressed_size;
        outfile_.write(reinterpret_cast<const char *>(host_bfr.get() + ck->ck_stat_size), ck->compressed_size);
        if (ck->ck_stat_size != 0) {
          md.row_groups[r].columns[i].meta_data.statistics_blob.resize(ck->ck_stat_size);
          memcpy(md.row_groups[r].columns[i].meta_data.statistics_blob.data(), host_bfr.get(), ck->ck_stat_size);
        }
        current_chunk_offset += ck->compressed_size;
      }
    }
  }

  CompactProtocolWriter cpw(&buffer_);
  file_ender_s fendr;
  fendr.footer_len = (uint32_t)cpw.write(&md);
  fendr.magic = PARQUET_MAGIC;
  outfile_.write(reinterpret_cast<char *>(buffer_.data()), buffer_.size());
  outfile_.write(reinterpret_cast<char *>(&fendr), sizeof(fendr));
  outfile_.flush();
}

// Forward to implementation
writer::writer(std::string filepath, writer_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(filepath, options, mr)) {}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write_all(table_view const &table, const table_metadata *metadata, cudaStream_t stream) {
  _impl->write(table, metadata, stream);
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
