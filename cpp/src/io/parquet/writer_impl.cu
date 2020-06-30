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
parquet::Compression to_parquet_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return parquet::Compression::SNAPPY;
    case compression_type::NONE: return parquet::Compression::UNCOMPRESSED;
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
 * @brief Helper class that adds parquet-specific column info
 **/
class parquet_column_view {
 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   **/
  explicit parquet_column_view(size_t id,
                               column_view const &col,
                               const table_metadata *metadata,
                               cudaStream_t stream)
    : _id(id),
      _string_type(col.type().id() == type_id::STRING),
      _type_width(_string_type ? 0 : cudf::size_of(col.type())),
      _converted_type(ConvertedType::UNKNOWN),
      _ts_scale(0),
      _data_count(col.size()),
      _null_count(col.null_count()),
      _data(col.head<uint8_t>() + col.offset() * _type_width),
      _nulls(col.nullable() ? col.null_mask() : nullptr)
  {
    switch (col.type().id()) {
      case cudf::type_id::INT8:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::INT_8;
        _stats_dtype    = statistics_dtype::dtype_int8;
        break;
      case cudf::type_id::INT16:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::INT_16;
        _stats_dtype    = statistics_dtype::dtype_int16;
        break;
      case cudf::type_id::INT32:
        _physical_type = Type::INT32;
        _stats_dtype   = statistics_dtype::dtype_int32;
        break;
      case cudf::type_id::INT64:
        _physical_type = Type::INT64;
        _stats_dtype   = statistics_dtype::dtype_int64;
        break;
      case cudf::type_id::UINT8:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::UINT_8;
        _stats_dtype    = statistics_dtype::dtype_int8;
        break;
      case cudf::type_id::UINT16:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::UINT_16;
        _stats_dtype    = statistics_dtype::dtype_int16;
        break;
      case cudf::type_id::UINT32:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::UINT_32;
        _stats_dtype    = statistics_dtype::dtype_int32;
        break;
      case cudf::type_id::UINT64:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::UINT_64;
        _stats_dtype    = statistics_dtype::dtype_int64;
        break;
      case cudf::type_id::FLOAT32:
        _physical_type = Type::FLOAT;
        _stats_dtype   = statistics_dtype::dtype_float32;
        break;
      case cudf::type_id::FLOAT64:
        _physical_type = Type::DOUBLE;
        _stats_dtype   = statistics_dtype::dtype_float64;
        break;
      case cudf::type_id::BOOL8:
        _physical_type = Type::BOOLEAN;
        _stats_dtype   = statistics_dtype::dtype_bool;
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::DATE;
        _stats_dtype    = statistics_dtype::dtype_int32;
        break;
      case cudf::type_id::TIMESTAMP_SECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MILLIS;
        _stats_dtype    = statistics_dtype::dtype_timestamp64;
        _ts_scale       = 1000;
        break;
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MILLIS;
        _stats_dtype    = statistics_dtype::dtype_timestamp64;
        break;
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MICROS;
        _stats_dtype    = statistics_dtype::dtype_timestamp64;
        break;
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIMESTAMP_MICROS;
        _stats_dtype    = statistics_dtype::dtype_timestamp64;
        _ts_scale       = -1000;
        break;
      case cudf::type_id::STRING:
        _physical_type = Type::BYTE_ARRAY;
        //_converted_type = ConvertedType::UTF8; // TBD
        _stats_dtype = statistics_dtype::dtype_string;
        break;
      default:
        _physical_type = UNDEFINED_TYPE;
        _stats_dtype   = dtype_none;
        break;
    }
    if (_string_type && _data_count > 0) {
      strings_column_view view{col};
      _indexes = rmm::device_buffer(_data_count * sizeof(gpu::nvstrdesc_s), stream);
      stringdata_to_nvstrdesc<<<((_data_count - 1) >> 8) + 1, 256, 0, stream>>>(
        reinterpret_cast<gpu::nvstrdesc_s *>(_indexes.data()),
        view.offsets().data<size_type>(),
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
  size_t type_width() const noexcept { return _type_width; }
  size_t data_count() const noexcept { return _data_count; }
  size_t null_count() const noexcept { return _null_count; }
  bool nullable() const noexcept { return (_nulls != nullptr); }
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
  void alloc_dictionary(size_t max_num_rows)
  {
    _dict_data.resize(max_num_rows);
    _dict_index.resize(max_num_rows);
  }
  bool check_dictionary_used()
  {
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
  size_t _id        = 0;
  bool _string_type = false;

  size_t _type_width     = 0;
  size_t _data_count     = 0;
  size_t _null_count     = 0;
  void const *_data      = nullptr;
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

void writer::impl::init_page_fragments(hostdevice_vector<gpu::PageFragment> &frag,
                                       hostdevice_vector<gpu::EncColumnDesc> &col_desc,
                                       uint32_t num_columns,
                                       uint32_t num_fragments,
                                       uint32_t num_rows,
                                       uint32_t fragment_size,
                                       cudaStream_t stream)
{
  CUDA_TRY(cudaMemcpyAsync(col_desc.device_ptr(),
                           col_desc.host_ptr(),
                           col_desc.memory_size(),
                           cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::InitPageFragments(frag.device_ptr(),
                                  col_desc.device_ptr(),
                                  num_fragments,
                                  num_columns,
                                  fragment_size,
                                  num_rows,
                                  stream));
  CUDA_TRY(cudaMemcpyAsync(
    frag.host_ptr(), frag.device_ptr(), frag.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

void writer::impl::gather_fragment_statistics(statistics_chunk *frag_stats_chunk,
                                              hostdevice_vector<gpu::PageFragment> &frag,
                                              hostdevice_vector<gpu::EncColumnDesc> &col_desc,
                                              uint32_t num_columns,
                                              uint32_t num_fragments,
                                              uint32_t fragment_size,
                                              cudaStream_t stream)
{
  rmm::device_vector<statistics_group> frag_stats_group(num_fragments * num_columns);

  CUDA_TRY(gpu::InitFragmentStatistics(frag_stats_group.data().get(),
                                       frag.device_ptr(),
                                       col_desc.device_ptr(),
                                       num_fragments,
                                       num_columns,
                                       fragment_size,
                                       stream));
  CUDA_TRY(GatherColumnStatistics(
    frag_stats_chunk, frag_stats_group.data().get(), num_fragments * num_columns, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

void writer::impl::build_chunk_dictionaries(hostdevice_vector<gpu::EncColumnChunk> &chunks,
                                            hostdevice_vector<gpu::EncColumnDesc> &col_desc,
                                            uint32_t num_rowgroups,
                                            uint32_t num_columns,
                                            uint32_t num_dictionaries,
                                            cudaStream_t stream)
{
  size_t dict_scratch_size = (size_t)num_dictionaries * gpu::kDictScratchSize;
  rmm::device_vector<uint32_t> dict_scratch(dict_scratch_size / sizeof(uint32_t));
  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::BuildChunkDictionaries(chunks.device_ptr(),
                                       dict_scratch.data().get(),
                                       dict_scratch_size,
                                       num_rowgroups * num_columns,
                                       stream));
  CUDA_TRY(gpu::InitEncoderPages(chunks.device_ptr(),
                                 nullptr,
                                 col_desc.device_ptr(),
                                 num_rowgroups,
                                 num_columns,
                                 nullptr,
                                 nullptr,
                                 stream));
  CUDA_TRY(cudaMemcpyAsync(
    chunks.host_ptr(), chunks.device_ptr(), chunks.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

void writer::impl::init_encoder_pages(hostdevice_vector<gpu::EncColumnChunk> &chunks,
                                      hostdevice_vector<gpu::EncColumnDesc> &col_desc,
                                      gpu::EncPage *pages,
                                      statistics_chunk *page_stats,
                                      statistics_chunk *frag_stats,
                                      uint32_t num_rowgroups,
                                      uint32_t num_columns,
                                      uint32_t num_pages,
                                      uint32_t num_stats_bfr,
                                      cudaStream_t stream)
{
  rmm::device_vector<statistics_merge_group> page_stats_mrg(num_stats_bfr);
  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(InitEncoderPages(
    chunks.device_ptr(),
    pages,
    col_desc.device_ptr(),
    num_rowgroups,
    num_columns,
    (num_stats_bfr) ? page_stats_mrg.data().get() : nullptr,
    (num_stats_bfr > num_pages) ? page_stats_mrg.data().get() + num_pages : nullptr,
    stream));
  if (num_stats_bfr > 0) {
    CUDA_TRY(MergeColumnStatistics(
      page_stats, frag_stats, page_stats_mrg.data().get(), num_pages, stream));
    if (num_stats_bfr > num_pages) {
      CUDA_TRY(MergeColumnStatistics(page_stats + num_pages,
                                     page_stats,
                                     page_stats_mrg.data().get() + num_pages,
                                     num_stats_bfr - num_pages,
                                     stream));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));
}

void writer::impl::encode_pages(hostdevice_vector<gpu::EncColumnChunk> &chunks,
                                gpu::EncPage *pages,
                                uint32_t num_columns,
                                uint32_t pages_in_batch,
                                uint32_t first_page_in_batch,
                                uint32_t rowgroups_in_batch,
                                uint32_t first_rowgroup,
                                gpu_inflate_input_s *comp_in,
                                gpu_inflate_status_s *comp_out,
                                const statistics_chunk *page_stats,
                                const statistics_chunk *chunk_stats,
                                cudaStream_t stream)
{
  CUDA_TRY(gpu::EncodePages(
    pages, chunks.device_ptr(), pages_in_batch, first_page_in_batch, comp_in, comp_out, stream));
  switch (compression_) {
    case parquet::Compression::SNAPPY:
      CUDA_TRY(gpu_snap(comp_in, comp_out, pages_in_batch, stream));
      break;
    default: break;
  }
  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the
  // chunk-level
  CUDA_TRY(DecideCompression(chunks.device_ptr() + first_rowgroup * num_columns,
                             pages,
                             rowgroups_in_batch * num_columns,
                             first_page_in_batch,
                             comp_out,
                             stream));
  CUDA_TRY(EncodePageHeaders(pages,
                             chunks.device_ptr(),
                             pages_in_batch,
                             first_page_in_batch,
                             comp_out,
                             page_stats,
                             chunk_stats,
                             stream));
  CUDA_TRY(GatherPages(chunks.device_ptr() + first_rowgroup * num_columns,
                       pages,
                       rowgroups_in_batch * num_columns,
                       stream));
  CUDA_TRY(cudaMemcpyAsync(&chunks[first_rowgroup * num_columns],
                           chunks.device_ptr() + first_rowgroup * num_columns,
                           rowgroups_in_batch * num_columns * sizeof(gpu::EncColumnChunk),
                           cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   writer_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _mr(mr),
    compression_(to_parquet_compression(options.compression)),
    stats_granularity_(options.stats_granularity),
    out_sink_(std::move(sink))
{
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::write(table_view const &table,
                                                          const table_metadata *metadata,
                                                          bool return_filemetadata,
                                                          const std::string &metadata_out_file_path,
                                                          cudaStream_t stream)
{
  pq_chunked_state state{metadata, SingleWriteMode::YES, stream};

  write_chunked_begin(state);
  write_chunked(table, state);
  return write_chunked_end(state, return_filemetadata, metadata_out_file_path);
}

void writer::impl::write_chunked_begin(pq_chunked_state &state)
{
  // Write file header
  file_header_s fhdr;
  fhdr.magic = PARQUET_MAGIC;
  out_sink_->host_write(&fhdr, sizeof(fhdr));
  state.current_chunk_offset = sizeof(file_header_s);
}

void writer::impl::write_chunked(table_view const &table, pq_chunked_state &state)
{
  size_type num_columns = table.num_columns();
  size_type num_rows    = 0;

  // Wrapper around cudf columns to attach parquet-specific type info.
  // Note : I wish we could do this in the begin() function but since the
  // metadata is optional we would have no way of knowing how many columns
  // we actually have.
  std::vector<parquet_column_view> parquet_columns;
  parquet_columns.reserve(num_columns);  // Avoids unnecessary re-allocation
  for (auto it = table.begin(); it < table.end(); ++it) {
    const auto col        = *it;
    const auto current_id = parquet_columns.size();

    num_rows = std::max<uint32_t>(num_rows, col.size());
    parquet_columns.emplace_back(current_id, col, state.user_metadata, state.stream);
  }

  if (state.user_metadata_with_nullability.column_nullable.size() > 0) {
    CUDF_EXPECTS(state.user_metadata_with_nullability.column_nullable.size() ==
                   static_cast<size_t>(num_columns),
                 "When passing values in user_metadata_with_nullability, data for all columns must "
                 "be specified");
  }

  // first call. setup metadata. num_rows will get incremented as write_chunked is
  // called multiple times.
  if (state.md.version == 0) {
    state.md.version  = 1;
    state.md.num_rows = num_rows;
    state.md.schema.resize(1 + num_columns);
    state.md.schema[0].type            = UNDEFINED_TYPE;
    state.md.schema[0].repetition_type = NO_REPETITION_TYPE;
    state.md.schema[0].name            = "schema";
    state.md.schema[0].num_children    = num_columns;
    state.md.column_order_listsize =
      (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_columns : 0;
    if (state.user_metadata != nullptr) {
      for (auto it = state.user_metadata->user_data.begin();
           it != state.user_metadata->user_data.end();
           it++) {
        state.md.key_value_metadata.push_back({it->first, it->second});
      }
    }
    for (auto i = 0; i < num_columns; i++) {
      auto &col = parquet_columns[i];
      // Column metadata
      state.md.schema[1 + i].type           = col.physical_type();
      state.md.schema[1 + i].converted_type = col.converted_type();
      // because the repetition type is global (in the sense of, not per-rowgroup or per
      // write_chunked() call) we cannot know up front if the user is going to end up passing tables
      // with nulls/no nulls in the multiple write_chunked() case.  so we'll do some special
      // handling.
      //
      // if the user is explicitly saying "I am only calling this once", fall back to the original
      // behavior and assume the columns in this one table tell us everything we need to know.
      if (state.single_write_mode) {
        state.md.schema[1 + i].repetition_type =
          (col.nullable() || col.data_count() < (size_t)num_rows) ? OPTIONAL : REQUIRED;
      }
      // otherwise, if the user is explicitly telling us global information about all the tables
      // that will ever get passed in
      else if (state.user_metadata_with_nullability.column_nullable.size() > 0) {
        state.md.schema[1 + i].repetition_type =
          state.user_metadata_with_nullability.column_nullable[i] ? OPTIONAL : REQUIRED;
      }
      // otherwise assume the worst case.
      else {
        state.md.schema[1 + i].repetition_type = OPTIONAL;
      }
      state.md.schema[1 + i].name         = col.name();
      state.md.schema[1 + i].num_children = 0;  // Leaf node
    }
  } else {
    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(state.md.schema[0].num_children == num_columns,
                 "Mismatch in table structure between multiple calls to write_chunked");
    for (auto i = 0; i < num_columns; i++) {
      auto &col = parquet_columns[i];
      CUDF_EXPECTS(state.md.schema[1 + i].type == col.physical_type(),
                   "Mismatch in column types between multiple calls to write_chunked");
    }

    // increment num rows
    state.md.num_rows += num_rows;
  }

  // Initialize column description
  hostdevice_vector<gpu::EncColumnDesc> col_desc(num_columns);

  // setup gpu column description.
  // applicable to only this _write_chunked() call
  for (auto i = 0; i < num_columns; i++) {
    auto &col = parquet_columns[i];
    // GPU column description
    auto *desc             = &col_desc[i];
    desc->column_data_base = col.data();
    desc->valid_map_base   = col.nulls();
    desc->stats_dtype      = col.stats_type();
    desc->ts_scale         = col.ts_scale();
    if (state.md.schema[1 + i].type != BOOLEAN && state.md.schema[1 + i].type != UNDEFINED_TYPE) {
      col.alloc_dictionary(num_rows);
      desc->dict_index = col.get_dict_index();
      desc->dict_data  = col.get_dict_data();
    } else {
      desc->dict_data  = nullptr;
      desc->dict_index = nullptr;
    }
    desc->num_rows       = col.data_count();
    desc->physical_type  = static_cast<uint8_t>(state.md.schema[1 + i].type);
    desc->converted_type = static_cast<uint8_t>(state.md.schema[1 + i].converted_type);
    desc->level_bits     = (state.md.schema[1 + i].repetition_type == OPTIONAL) ? 1 : 0;
  }

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings will start producing
  // fragments larger than the desired page size -> TODO: keep track of the max fragment size, and
  // iteratively reduce this value if the largest fragment exceeds the max page size limit (we
  // ideally want the page size to be below 1MB so as to have enough pages to get good
  // compression/decompression performance).
  uint32_t fragment_size = 5000;
  uint32_t num_fragments = (uint32_t)((num_rows + fragment_size - 1) / fragment_size);
  hostdevice_vector<gpu::PageFragment> fragments(num_columns * num_fragments);
  if (fragments.size() != 0) {
    init_page_fragments(
      fragments, col_desc, num_columns, num_fragments, num_rows, fragment_size, state.stream);
  }

  size_t global_rowgroup_base = state.md.row_groups.size();

  // Decide row group boundaries based on uncompressed data size
  size_t rowgroup_size   = 0;
  uint32_t num_rowgroups = 0;
  for (uint32_t f = 0, global_r = global_rowgroup_base, rowgroup_start = 0; f < num_fragments;
       f++) {
    size_t fragment_data_size = 0;
    for (auto i = 0; i < num_columns; i++) {
      fragment_data_size += fragments[i * num_fragments + f].fragment_data_size;
    }
    if (f > rowgroup_start && (rowgroup_size + fragment_data_size > max_rowgroup_size_ ||
                               (f + 1 - rowgroup_start) * fragment_size > max_rowgroup_rows_)) {
      // update schema
      state.md.row_groups.resize(state.md.row_groups.size() + 1);
      state.md.row_groups[global_r++].num_rows = (f - rowgroup_start) * fragment_size;
      num_rowgroups++;
      rowgroup_start = f;
      rowgroup_size  = 0;
    }
    rowgroup_size += fragment_data_size;
    if (f + 1 == num_fragments) {
      // update schema
      state.md.row_groups.resize(state.md.row_groups.size() + 1);
      state.md.row_groups[global_r++].num_rows = num_rows - rowgroup_start * fragment_size;
      num_rowgroups++;
    }
  }

  // Allocate column chunks and gather fragment statistics
  rmm::device_vector<statistics_chunk> frag_stats;
  if (stats_granularity_ != statistics_freq::STATISTICS_NONE) {
    frag_stats.resize(num_fragments * num_columns);
    if (frag_stats.size() != 0) {
      gather_fragment_statistics(frag_stats.data().get(),
                                 fragments,
                                 col_desc,
                                 num_columns,
                                 num_fragments,
                                 fragment_size,
                                 state.stream);
    }
  }

  // Initialize row groups and column chunks
  uint32_t num_chunks = num_rowgroups * num_columns;
  hostdevice_vector<gpu::EncColumnChunk> chunks(num_chunks);
  uint32_t num_dictionaries = 0;
  for (uint32_t r = 0, global_r = global_rowgroup_base, f = 0, start_row = 0; r < num_rowgroups;
       r++, global_r++) {
    uint32_t fragments_in_chunk =
      (uint32_t)((state.md.row_groups[global_r].num_rows + fragment_size - 1) / fragment_size);
    state.md.row_groups[global_r].total_byte_size = 0;
    state.md.row_groups[global_r].columns.resize(num_columns);
    for (int i = 0; i < num_columns; i++) {
      gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
      bool dict_enable        = false;

      ck->col_desc         = col_desc.device_ptr() + i;
      ck->uncompressed_bfr = nullptr;
      ck->compressed_bfr   = nullptr;
      ck->bfr_size         = 0;
      ck->compressed_size  = 0;
      ck->fragments        = fragments.device_ptr() + i * num_fragments + f;
      ck->stats =
        (frag_stats.size() != 0) ? frag_stats.data().get() + i * num_fragments + f : nullptr;
      ck->start_row      = start_row;
      ck->num_rows       = (uint32_t)state.md.row_groups[global_r].num_rows;
      ck->first_fragment = i * num_fragments + f;
      ck->first_page     = 0;
      ck->num_pages      = 0;
      ck->is_compressed  = 0;
      ck->dictionary_id  = num_dictionaries;
      ck->ck_stat_size   = 0;
      if (col_desc[i].dict_data) {
        const gpu::PageFragment *ck_frag = &fragments[i * num_fragments + f];
        size_t plain_size                = 0;
        size_t dict_size                 = 1;
        uint32_t num_dict_vals           = 0;
        for (uint32_t j = 0; j < fragments_in_chunk && num_dict_vals < 65536; j++) {
          plain_size += ck_frag[j].fragment_data_size;
          dict_size +=
            ck_frag[j].dict_data_size + ((num_dict_vals > 256) ? 2 : 1) * ck_frag[j].non_nulls;
          num_dict_vals += ck_frag[j].num_dict_vals;
        }
        if (dict_size < plain_size) {
          parquet_columns[i].use_dictionary(true);
          dict_enable = true;
          num_dictionaries++;
        }
      }
      ck->has_dictionary                                           = dict_enable;
      state.md.row_groups[global_r].columns[i].meta_data.type      = state.md.schema[1 + i].type;
      state.md.row_groups[global_r].columns[i].meta_data.encodings = {PLAIN, RLE};
      if (dict_enable) {
        state.md.row_groups[global_r].columns[i].meta_data.encodings.push_back(PLAIN_DICTIONARY);
      }
      state.md.row_groups[global_r].columns[i].meta_data.path_in_schema = {
        state.md.schema[1 + i].name};
      state.md.row_groups[global_r].columns[i].meta_data.codec = UNCOMPRESSED;
      state.md.row_groups[global_r].columns[i].meta_data.num_values =
        state.md.row_groups[global_r].num_rows;
    }
    f += fragments_in_chunk;
    start_row += (uint32_t)state.md.row_groups[global_r].num_rows;
  }

  // Free unused dictionaries
  for (auto &col : parquet_columns) { col.check_dictionary_used(); }

  // Build chunk dictionaries and count pages
  if (num_chunks != 0) {
    build_chunk_dictionaries(
      chunks, col_desc, num_rowgroups, num_columns, num_dictionaries, state.stream);
  }

  // Initialize batches of rowgroups to encode (mainly to limit peak memory usage)
  std::vector<uint32_t> batch_list;
  uint32_t num_pages          = 0;
  size_t max_bytes_in_batch   = 1024 * 1024 * 1024;  // 1GB - TBD: Tune this
  size_t max_uncomp_bfr_size  = 0;
  size_t max_chunk_bfr_size   = 0;
  uint32_t max_pages_in_batch = 0;
  size_t bytes_in_batch       = 0;
  for (uint32_t r = 0, groups_in_batch = 0, pages_in_batch = 0; r <= num_rowgroups; r++) {
    size_t rowgroup_size = 0;
    if (r < num_rowgroups) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        ck->first_page          = num_pages;
        num_pages += ck->num_pages;
        pages_in_batch += ck->num_pages;
        rowgroup_size += ck->bfr_size;
        max_chunk_bfr_size =
          std::max(max_chunk_bfr_size, (size_t)std::max(ck->bfr_size, ck->compressed_size));
      }
    }
    // TBD: We may want to also shorten the batch if we have enough pages (not just based on size)
    if ((r == num_rowgroups) ||
        (groups_in_batch != 0 && bytes_in_batch + rowgroup_size > max_bytes_in_batch)) {
      max_uncomp_bfr_size = std::max(max_uncomp_bfr_size, bytes_in_batch);
      max_pages_in_batch  = std::max(max_pages_in_batch, pages_in_batch);
      if (groups_in_batch != 0) {
        batch_list.push_back(groups_in_batch);
        groups_in_batch = 0;
      }
      bytes_in_batch = 0;
      pages_in_batch = 0;
    }
    bytes_in_batch += rowgroup_size;
    groups_in_batch++;
  }

  // Initialize data pointers in batch
  size_t max_comp_bfr_size =
    (compression_ != parquet::Compression::UNCOMPRESSED)
      ? gpu::GetMaxCompressedBfrSize(max_uncomp_bfr_size, max_pages_in_batch)
      : 0;
  uint32_t max_comp_pages =
    (compression_ != parquet::Compression::UNCOMPRESSED) ? max_pages_in_batch : 0;
  uint32_t num_stats_bfr =
    (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;
  rmm::device_buffer uncomp_bfr(max_uncomp_bfr_size, state.stream);
  rmm::device_buffer comp_bfr(max_comp_bfr_size, state.stream);
  rmm::device_vector<gpu_inflate_input_s> comp_in(max_comp_pages);
  rmm::device_vector<gpu_inflate_status_s> comp_out(max_comp_pages);
  rmm::device_vector<gpu::EncPage> pages(num_pages);
  rmm::device_vector<statistics_chunk> page_stats(num_stats_bfr);
  for (uint32_t b = 0, r = 0; b < (uint32_t)batch_list.size(); b++) {
    uint8_t *bfr   = reinterpret_cast<uint8_t *>(uncomp_bfr.data());
    uint8_t *bfr_c = reinterpret_cast<uint8_t *>(comp_bfr.data());
    for (uint32_t j = 0; j < batch_list[b]; j++, r++) {
      for (int i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        ck->uncompressed_bfr    = bfr;
        ck->compressed_bfr      = bfr_c;
        bfr += ck->bfr_size;
        bfr_c += ck->compressed_size;
      }
    }
  }

  if (num_pages != 0) {
    init_encoder_pages(chunks,
                       col_desc,
                       pages.data().get(),
                       (num_stats_bfr) ? page_stats.data().get() : nullptr,
                       (num_stats_bfr) ? frag_stats.data().get() : nullptr,
                       num_rowgroups,
                       num_columns,
                       num_pages,
                       num_stats_bfr,
                       state.stream);
  }

  auto host_bfr = [&]() {
    // if the writer supports device_write(), we don't need this scratch space
    if (out_sink_->supports_device_write()) {
      return pinned_buffer<uint8_t>{nullptr, cudaFreeHost};
    } else {
      return pinned_buffer<uint8_t>{[](size_t size) {
                                      uint8_t *ptr = nullptr;
                                      CUDA_TRY(cudaMallocHost(&ptr, size));
                                      return ptr;
                                    }(max_chunk_bfr_size),
                                    cudaFreeHost};
    }
  }();

  // Encode row groups in batches
  for (uint32_t b = 0, r = 0, global_r = global_rowgroup_base; b < (uint32_t)batch_list.size();
       b++) {
    // Count pages in this batch
    uint32_t rnext               = r + batch_list[b];
    uint32_t first_page_in_batch = chunks[r * num_columns].first_page;
    uint32_t first_page_in_next_batch =
      (rnext < num_rowgroups) ? chunks[rnext * num_columns].first_page : num_pages;
    uint32_t pages_in_batch = first_page_in_next_batch - first_page_in_batch;
    encode_pages(
      chunks,
      pages.data().get(),
      num_columns,
      pages_in_batch,
      first_page_in_batch,
      batch_list[b],
      r,
      comp_in.data().get(),
      comp_out.data().get(),
      (stats_granularity_ == statistics_freq::STATISTICS_PAGE) ? page_stats.data().get() : nullptr,
      (stats_granularity_ != statistics_freq::STATISTICS_NONE) ? page_stats.data().get() + num_pages
                                                               : nullptr,
      state.stream);
    for (; r < rnext; r++, global_r++) {
      for (auto i = 0; i < num_columns; i++) {
        gpu::EncColumnChunk *ck = &chunks[r * num_columns + i];
        uint8_t *dev_bfr;
        if (ck->is_compressed) {
          state.md.row_groups[global_r].columns[i].meta_data.codec = compression_;
          dev_bfr                                                  = ck->compressed_bfr;
        } else {
          dev_bfr = ck->uncompressed_bfr;
        }

        if (out_sink_->supports_device_write()) {
          // let the writer do what it wants to retrieve the data from the gpu.
          out_sink_->device_write(dev_bfr + ck->ck_stat_size, ck->compressed_size, state.stream);
          // we still need to do a (much smaller) memcpy for the statistics.
          if (ck->ck_stat_size != 0) {
            state.md.row_groups[global_r].columns[i].meta_data.statistics_blob.resize(
              ck->ck_stat_size);
            CUDA_TRY(cudaMemcpyAsync(
              state.md.row_groups[global_r].columns[i].meta_data.statistics_blob.data(),
              dev_bfr,
              ck->ck_stat_size,
              cudaMemcpyDeviceToHost,
              state.stream));
            CUDA_TRY(cudaStreamSynchronize(state.stream));
          }
        } else {
          // copy the full data
          CUDA_TRY(cudaMemcpyAsync(host_bfr.get(),
                                   dev_bfr,
                                   ck->ck_stat_size + ck->compressed_size,
                                   cudaMemcpyDeviceToHost,
                                   state.stream));
          CUDA_TRY(cudaStreamSynchronize(state.stream));
          out_sink_->host_write(host_bfr.get() + ck->ck_stat_size, ck->compressed_size);
          if (ck->ck_stat_size != 0) {
            state.md.row_groups[global_r].columns[i].meta_data.statistics_blob.resize(
              ck->ck_stat_size);
            memcpy(state.md.row_groups[global_r].columns[i].meta_data.statistics_blob.data(),
                   host_bfr.get(),
                   ck->ck_stat_size);
          }
        }
        state.md.row_groups[global_r].total_byte_size += ck->compressed_size;
        state.md.row_groups[global_r].columns[i].meta_data.data_page_offset =
          state.current_chunk_offset + ((ck->has_dictionary) ? ck->dictionary_size : 0);
        state.md.row_groups[global_r].columns[i].meta_data.dictionary_page_offset =
          (ck->has_dictionary) ? state.current_chunk_offset : 0;
        state.md.row_groups[global_r].columns[i].meta_data.total_uncompressed_size = ck->bfr_size;
        state.md.row_groups[global_r].columns[i].meta_data.total_compressed_size =
          ck->compressed_size;
        state.current_chunk_offset += ck->compressed_size;
      }
    }
  }
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::write_chunked_end(
  pq_chunked_state &state, bool return_filemetadata, const std::string &metadata_out_file_path)
{
  CompactProtocolWriter cpw(&buffer_);
  file_ender_s fendr;
  buffer_.resize(0);
  fendr.footer_len = static_cast<uint32_t>(cpw.write(&state.md));
  fendr.magic      = PARQUET_MAGIC;
  out_sink_->host_write(buffer_.data(), buffer_.size());
  out_sink_->host_write(&fendr, sizeof(fendr));
  out_sink_->flush();

  // Optionally output raw file metadata with the specified column chunk file path
  if (return_filemetadata) {
    file_header_s fhdr = {PARQUET_MAGIC};
    buffer_.resize(0);
    buffer_.insert(buffer_.end(),
                   reinterpret_cast<const uint8_t *>(&fhdr),
                   reinterpret_cast<const uint8_t *>(&fhdr) + sizeof(fhdr));
    for (auto &rowgroup : state.md.row_groups) {
      for (auto &col : rowgroup.columns) { col.file_path = metadata_out_file_path; }
    }
    fendr.footer_len = static_cast<uint32_t>(cpw.write(&state.md));
    buffer_.insert(buffer_.end(),
                   reinterpret_cast<const uint8_t *>(&fendr),
                   reinterpret_cast<const uint8_t *>(&fendr) + sizeof(fendr));
    return std::make_unique<std::vector<uint8_t>>(std::move(buffer_));
  } else {
    return {nullptr};
  }
}

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               writer_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mr))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::write_all(table_view const &table,
                                                        const table_metadata *metadata,
                                                        bool return_filemetadata,
                                                        const std::string metadata_out_file_path,
                                                        cudaStream_t stream)
{
  return _impl->write(table, metadata, return_filemetadata, metadata_out_file_path, stream);
}

// Forward to implementation
void writer::write_chunked_begin(pq_chunked_state &state)
{
  return _impl->write_chunked_begin(state);
}

// Forward to implementation
void writer::write_chunked(table_view const &table, pq_chunked_state &state)
{
  _impl->write_chunked(table, state);
}

// Forward to implementation
void writer::write_chunked_end(pq_chunked_state &state) { _impl->write_chunked_end(state); }

std::unique_ptr<std::vector<uint8_t>> writer::merge_rowgroup_metadata(
  const std::vector<std::unique_ptr<std::vector<uint8_t>>> &metadata_list)
{
  std::vector<uint8_t> output;
  CompactProtocolWriter cpw(&output);
  FileMetaData md;

  md.row_groups.reserve(metadata_list.size());
  for (const auto &blob : metadata_list) {
    CompactProtocolReader cpreader(
      blob.get()->data(),
      std::max<size_t>(blob.get()->size(), sizeof(file_ender_s)) - sizeof(file_ender_s));
    cpreader.skip_bytes(sizeof(file_header_s));  // Skip over file header
    if (md.num_rows == 0) {
      cpreader.read(&md);
    } else {
      FileMetaData tmp;
      cpreader.read(&tmp);
      md.row_groups.insert(md.row_groups.end(),
                           std::make_move_iterator(tmp.row_groups.begin()),
                           std::make_move_iterator(tmp.row_groups.end()));
      md.num_rows += tmp.num_rows;
    }
  }
  // Reader doesn't currently populate column_order, so infer it here
  if (md.row_groups.size() != 0) {
    uint32_t num_columns = static_cast<uint32_t>(md.row_groups[0].columns.size());
    md.column_order_listsize =
      (num_columns > 0 && md.row_groups[0].columns[0].meta_data.statistics_blob.size())
        ? num_columns
        : 0;
  }
  // Thrift-encode the resulting output
  file_header_s fhdr;
  file_ender_s fendr;
  fhdr.magic = PARQUET_MAGIC;
  output.insert(output.end(),
                reinterpret_cast<const uint8_t *>(&fhdr),
                reinterpret_cast<const uint8_t *>(&fhdr) + sizeof(fhdr));
  fendr.footer_len = static_cast<uint32_t>(cpw.write(&md));
  fendr.magic      = PARQUET_MAGIC;
  output.insert(output.end(),
                reinterpret_cast<const uint8_t *>(&fendr),
                reinterpret_cast<const uint8_t *>(&fendr) + sizeof(fendr));
  return std::make_unique<std::vector<uint8_t>>(std::move(output));
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
