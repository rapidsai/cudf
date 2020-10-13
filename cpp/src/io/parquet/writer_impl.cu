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

#include <io/parquet/compact_protocol_writer.hpp>
#include "writer_impl.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

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

/**
 * @brief Get the leaf column
 *
 * Returns the dtype of the leaf column when `col` is a list column.
 */
column_view get_leaf_col(column_view col)
{
  column_view curr_col = col;
  while (curr_col.type().id() == type_id::LIST) { curr_col = lists_column_view{curr_col}.child(); }
  return curr_col;
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
    : _col(col),
      _leaf_col(get_leaf_col(col)),
      _id(id),
      _string_type(_leaf_col.type().id() == type_id::STRING),
      _list_type(col.type().id() == type_id::LIST),
      _type_width((_string_type || _list_type) ? 0 : cudf::size_of(col.type())),
      _row_count(col.size()),
      _null_count(_leaf_col.null_count()),
      _data(col.head<uint8_t>() + col.offset() * _type_width),
      _nulls(_leaf_col.nullable() ? _leaf_col.null_mask() : nullptr),
      _converted_type(ConvertedType::UNKNOWN),
      _ts_scale(0),
      _dremel_offsets(0, stream),
      _rep_level(0, stream),
      _def_level(0, stream)
  {
    switch (_leaf_col.type().id()) {
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
      // unsupported outside cudf for parquet 1.0.
      case cudf::type_id::DURATION_DAYS:
        _physical_type  = Type::INT32;
        _converted_type = ConvertedType::TIME_MILLIS;
        _stats_dtype    = statistics_dtype::dtype_int64;
        break;
      case cudf::type_id::DURATION_SECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIME_MILLIS;
        _stats_dtype    = statistics_dtype::dtype_int64;
        _ts_scale       = 1000;
        break;
      case cudf::type_id::DURATION_MILLISECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIME_MILLIS;
        _stats_dtype    = statistics_dtype::dtype_int64;
        break;
      case cudf::type_id::DURATION_MICROSECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIME_MICROS;
        _stats_dtype    = statistics_dtype::dtype_int64;
        break;
      // unsupported outside cudf for parquet 1.0.
      case cudf::type_id::DURATION_NANOSECONDS:
        _physical_type  = Type::INT64;
        _converted_type = ConvertedType::TIME_MICROS;
        _stats_dtype    = statistics_dtype::dtype_int64;
        _ts_scale       = -1000;  // negative value indicates division by absolute value
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
        _ts_scale       = -1000;  // negative value indicates division by absolute value
        break;
      case cudf::type_id::STRING:
        _physical_type  = Type::BYTE_ARRAY;
        _converted_type = ConvertedType::UTF8;
        _stats_dtype    = statistics_dtype::dtype_string;
        break;
      default:
        _physical_type = UNDEFINED_TYPE;
        _stats_dtype   = dtype_none;
        break;
    }
    size_type leaf_col_offset = col.offset();
    _data_count               = col.size();
    if (_list_type) {
      // Top level column's offsets are not applied to all children. Get the effective offset and
      // size of the leaf column
      // Calculate row offset into dremel data (repetition/definition values) and the respective
      // definition and repetition levels
      gpu::dremel_data dremel = gpu::get_dremel_data(col, stream);
      _dremel_offsets         = std::move(dremel.dremel_offsets);
      _rep_level              = std::move(dremel.rep_level);
      _def_level              = std::move(dremel.def_level);
      leaf_col_offset         = dremel.leaf_col_offset;
      _data_count             = dremel.leaf_data_size;

      _type_width = (is_fixed_width(_leaf_col.type())) ? cudf::size_of(_leaf_col.type()) : 0;
      _data       = (is_fixed_width(_leaf_col.type()))
                ? _leaf_col.head<uint8_t>() + leaf_col_offset * _type_width
                : nullptr;

      // Bring offset array to device
      column_view curr_col = col;
      std::vector<size_type const *> offsets_array;
      while (curr_col.type().id() == type_id::LIST) {
        lists_column_view list_col(curr_col);
        offsets_array.push_back(list_col.offsets().data<size_type>());
        curr_col = list_col.child();
      }
      _offsets_array = offsets_array;

      CUDA_TRY(cudaStreamSynchronize(stream));
    }
    if (_string_type && _data_count > 0) {
      strings_column_view view{_leaf_col};
      _indexes = rmm::device_buffer(_data_count * sizeof(gpu::nvstrdesc_s), stream);
      stringdata_to_nvstrdesc<<<((_data_count - 1) >> 8) + 1, 256, 0, stream>>>(
        reinterpret_cast<gpu::nvstrdesc_s *>(_indexes.data()),
        view.offsets().data<size_type>() + leaf_col_offset,
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
    _path_in_schema.push_back(_name);
  }

  auto is_string() const noexcept { return _string_type; }
  auto is_list() const noexcept { return _list_type; }
  size_t type_width() const noexcept { return _type_width; }
  size_t row_count() const noexcept { return _row_count; }
  size_t data_count() const noexcept { return _data_count; }
  size_t null_count() const noexcept { return _null_count; }
  bool nullable() const noexcept { return (_nulls != nullptr); }
  void const *data() const noexcept { return _data; }
  uint32_t const *nulls() const noexcept { return _nulls; }

  // List related data
  column_view cudf_col() const noexcept { return _col; }
  column_view leaf_col() const noexcept { return _leaf_col; }
  size_type const *const *nesting_offsets() const noexcept { return _offsets_array.data().get(); }
  size_type nesting_levels() const noexcept { return _offsets_array.size(); }
  size_type const *level_offsets() const noexcept { return _dremel_offsets.data(); }
  uint8_t const *repetition_levels() const noexcept { return _rep_level.data(); }
  uint8_t const *definition_levels() const noexcept { return _def_level.data(); }
  uint16_t max_def_level()
  {
    if (_max_def_level > -1) return _max_def_level;

    uint16_t num_def_level = 0;
    auto curr_col          = cudf_col();
    while (curr_col.type().id() == type_id::LIST) {
      // There is one definition level for each level of nesting and one for each level's
      // nullability. If the level is nullable, then it needs 2 definition levels to describe it.
      num_def_level += curr_col.nullable() ? 2 : 1;
      lists_column_view lcw(curr_col);
      curr_col = lcw.child();
    }
    // This is the leaf level. There is no further nesting.
    if (curr_col.nullable()) ++num_def_level;
    _max_def_level = num_def_level;
    return _max_def_level;
  }
  void set_def_level(uint16_t def_level) { _max_def_level = def_level; }

  auto name() const noexcept { return _name; }
  auto physical_type() const noexcept { return _physical_type; }
  auto converted_type() const noexcept { return _converted_type; }
  auto stats_type() const noexcept { return _stats_dtype; }
  int32_t ts_scale() const noexcept { return _ts_scale; }
  void set_path_in_schema(std::vector<std::string> path) { _path_in_schema = std::move(path); }
  auto get_path_in_schema() const noexcept { return _path_in_schema; }

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
  // cudf data column
  column_view _col;
  column_view _leaf_col;

  // Identifier within set of columns
  size_t _id        = 0;
  bool _string_type = false;
  bool _list_type   = false;

  size_t _type_width     = 0;
  size_t _row_count      = 0;
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
  std::vector<std::string> _path_in_schema;

  // Dictionary-related members
  bool _dictionary_used = false;
  rmm::device_vector<uint32_t> _dict_data;
  rmm::device_vector<uint32_t> _dict_index;

  // List-related members
  // TODO (dm): convert to uvector
  rmm::device_vector<size_type const *> _offsets_array;  ///< Array of pointers to offset columns at
                                                         ///< each level of nesting O(nesting depth)
  rmm::device_uvector<size_type>
    _dremel_offsets;  ///< For each row, the absolute offset into the repetition and definition
                      ///< level vectors. O(num rows)
  rmm::device_uvector<uint8_t> _rep_level;
  rmm::device_uvector<uint8_t> _def_level;
  size_type _max_def_level = -1;

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
                   parquet_writer_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _mr(mr),
    compression_(to_parquet_compression(options.get_compression())),
    stats_granularity_(options.get_stats_level()),
    out_sink_(std::move(sink))
{
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::write(
  table_view const &table,
  const table_metadata *metadata,
  bool return_filemetadata,
  const std::string &column_chunks_file_path,
  cudaStream_t stream)
{
  pq_chunked_state state{metadata, SingleWriteMode::YES, stream};

  write_chunked_begin(state);
  write_chunk(table, state);
  return write_chunked_end(state, return_filemetadata, column_chunks_file_path);
}

void writer::impl::write_chunked_begin(pq_chunked_state &state)
{
  // Write file header
  file_header_s fhdr;
  fhdr.magic = PARQUET_MAGIC;
  out_sink_->host_write(&fhdr, sizeof(fhdr));
  state.current_chunk_offset = sizeof(file_header_s);
}

void writer::impl::write_chunk(table_view const &table, pq_chunked_state &state)
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

  // first call. setup metadata. num_rows will get incremented as write_chunk is
  // called multiple times.
  // Calculate the sum of depths of all list columns
  size_type const list_col_depths = std::accumulate(
    parquet_columns.cbegin(), parquet_columns.cend(), 0, [](size_type sum, auto const &col) {
      return sum + col.nesting_levels();
    });

  if (state.md.version == 0) {
    state.md.version  = 1;
    state.md.num_rows = num_rows;
    // Each level of nesting requires two levels of Schema. The leaf level needs one schema element
    state.md.schema.reserve(1 + num_columns + list_col_depths * 2);
    SchemaElement root{};
    root.type            = UNDEFINED_TYPE;
    root.repetition_type = NO_REPETITION_TYPE;
    root.name            = "schema";
    root.num_children    = num_columns;
    state.md.schema.push_back(std::move(root));
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
      if (col.is_list()) {
        CUDF_EXPECTS(state.single_write_mode, "Chunked write for lists is not supported");
        size_type nesting_depth = col.nesting_levels();
        // Each level of nesting requires two levels of Schema. The leaf level needs one schema
        // element
        std::vector<SchemaElement> list_schema(nesting_depth * 2 + 1);
        column_view cudf_col = col.cudf_col();
        for (size_type j = 0; j < nesting_depth; j++) {
          // List schema is denoted by two levels for each nesting level and one final level for
          // leaf. The top level is the same name as the column name.
          // So e.g. List<List<int>> is denoted in the schema by
          // "col_name" : { "list" : { "element" : { "list" : { "element" } } } }
          auto const group_idx = 2 * j;
          auto const list_idx  = 2 * j + 1;

          list_schema[group_idx].name            = (j == 0) ? col.name() : "element";
          list_schema[group_idx].repetition_type = (cudf_col.nullable()) ? OPTIONAL : REQUIRED;
          list_schema[group_idx].converted_type  = ConvertedType::LIST;
          list_schema[group_idx].num_children    = 1;

          list_schema[list_idx].name            = "list";
          list_schema[list_idx].repetition_type = REPEATED;
          list_schema[list_idx].num_children    = 1;

          // Move on to next child
          lists_column_view lcw(cudf_col);
          cudf_col = lcw.child();
        }
        list_schema[nesting_depth * 2].name = "element";
        list_schema[nesting_depth * 2].repetition_type =
          col.leaf_col().nullable() ? OPTIONAL : REQUIRED;
        list_schema[nesting_depth * 2].type           = col.physical_type();
        list_schema[nesting_depth * 2].converted_type = col.converted_type();
        list_schema[nesting_depth * 2].num_children   = 0;

        std::vector<std::string> path_in_schema;
        std::transform(
          list_schema.cbegin(), list_schema.cend(), std::back_inserter(path_in_schema), [](auto s) {
            return s.name;
          });
        col.set_path_in_schema(path_in_schema);
        state.md.schema.insert(state.md.schema.end(), list_schema.begin(), list_schema.end());
      } else {
        SchemaElement col_schema{};
        // Column metadata
        col_schema.type           = col.physical_type();
        col_schema.converted_type = col.converted_type();
        // because the repetition type is global (in the sense of, not per-rowgroup or per
        // write_chunk() call) we cannot know up front if the user is going to end up passing tables
        // with nulls/no nulls in the multiple write_chunk() case.  so we'll do some special
        // handling.
        //
        // if the user is explicitly saying "I am only calling this once", fall back to the original
        // behavior and assume the columns in this one table tell us everything we need to know.
        if (state.single_write_mode) {
          col_schema.repetition_type =
            (col.nullable() || col.data_count() < (size_t)num_rows) ? OPTIONAL : REQUIRED;
        }
        // otherwise, if the user is explicitly telling us global information about all the tables
        // that will ever get passed in
        else if (state.user_metadata_with_nullability.column_nullable.size() > 0) {
          col_schema.repetition_type =
            state.user_metadata_with_nullability.column_nullable[i] ? OPTIONAL : REQUIRED;
          col.set_def_level((col_schema.repetition_type == OPTIONAL) ? 1 : 0);
        }
        // otherwise assume the worst case.
        else {
          col_schema.repetition_type = OPTIONAL;
          col.set_def_level(1);  // def level for OPTIONAL is 1, for REQUIRED is 0
        }
        col_schema.name         = col.name();
        col_schema.num_children = 0;  // Leaf node

        state.md.schema.push_back(std::move(col_schema));
      }
    }
  } else {
    // verify the user isn't passing mismatched tables
    // TODO (dm): Now needs to compare children of columns in case of list when we support chunked
    // write for it
    CUDF_EXPECTS(state.md.schema[0].num_children == num_columns,
                 "Mismatch in table structure between multiple calls to write_chunk");
    for (auto i = 0; i < num_columns; i++) {
      auto &col = parquet_columns[i];
      CUDF_EXPECTS(state.md.schema[1 + i].type == col.physical_type(),
                   "Mismatch in column types between multiple calls to write_chunk");
    }

    // increment num rows
    state.md.num_rows += num_rows;
  }

  // Initialize column description
  hostdevice_vector<gpu::EncColumnDesc> col_desc(num_columns);

  // setup gpu column description.
  // applicable to only this _write_chunk() call
  for (auto i = 0; i < num_columns; i++) {
    auto &col = parquet_columns[i];
    // GPU column description
    auto *desc             = &col_desc[i];
    *desc                  = gpu::EncColumnDesc{};  // Zero out all fields
    desc->column_data_base = col.data();
    desc->valid_map_base   = col.nulls();
    desc->stats_dtype      = col.stats_type();
    desc->ts_scale         = col.ts_scale();
    // TODO (dm): Enable dictionary for list after refactor
    if (col.physical_type() != BOOLEAN && col.physical_type() != UNDEFINED_TYPE && !col.is_list()) {
      col.alloc_dictionary(col.data_count());
      desc->dict_index = col.get_dict_index();
      desc->dict_data  = col.get_dict_data();
    }
    if (col.is_list()) {
      desc->nesting_offsets = col.nesting_offsets();
      desc->nesting_levels  = col.nesting_levels();
      desc->level_offsets   = col.level_offsets();
      desc->rep_values      = col.repetition_levels();
      desc->def_values      = col.definition_levels();
      auto count_bits       = [](uint16_t number) {
        int16_t nbits = 0;
        while (number > 0) {
          nbits++;
          number = number >> 1;
        }
        return nbits;
      };
      desc->level_bits = count_bits(col.nesting_levels()) << 4 | count_bits(col.max_def_level());
    } else {
      desc->level_bits = (state.md.schema[1 + i].repetition_type == OPTIONAL) ? 1 : 0;
    }
    desc->num_values     = col.data_count();
    desc->num_rows       = col.row_count();
    desc->physical_type  = static_cast<uint8_t>(col.physical_type());
    desc->converted_type = static_cast<uint8_t>(col.converted_type());
  }

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings will start producing
  // fragments larger than the desired page size -> TODO: keep track of the max fragment size, and
  // iteratively reduce this value if the largest fragment exceeds the max page size limit (we
  // ideally want the page size to be below 1MB so as to have enough pages to get good
  // compression/decompression performance).
  constexpr uint32_t fragment_size = 5000;
  static_assert(fragment_size <= MAX_PAGE_FRAGMENT_SIZE,
                "fragment size cannot be greater than MAX_PAGE_FRAGMENT_SIZE");

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
    // Replace with STL algorithm to transform and sum
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
      ck->num_values =
        std::accumulate(fragments.host_ptr(i * num_fragments + f),
                        fragments.host_ptr(i * num_fragments + f) + fragments_in_chunk,
                        0,
                        [](uint32_t l, auto r) { return l + r.num_values; });
      ck->first_page    = 0;
      ck->num_pages     = 0;
      ck->is_compressed = 0;
      ck->dictionary_id = num_dictionaries;
      ck->ck_stat_size  = 0;
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
      ck->has_dictionary                                      = dict_enable;
      state.md.row_groups[global_r].columns[i].meta_data.type = parquet_columns[i].physical_type();
      state.md.row_groups[global_r].columns[i].meta_data.encodings = {PLAIN, RLE};
      if (dict_enable) {
        state.md.row_groups[global_r].columns[i].meta_data.encodings.push_back(PLAIN_DICTIONARY);
      }
      state.md.row_groups[global_r].columns[i].meta_data.path_in_schema =
        parquet_columns[i].get_path_in_schema();
      state.md.row_groups[global_r].columns[i].meta_data.codec      = UNCOMPRESSED;
      state.md.row_groups[global_r].columns[i].meta_data.num_values = ck->num_values;
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
    uint8_t *bfr   = static_cast<uint8_t *>(uncomp_bfr.data());
    uint8_t *bfr_c = static_cast<uint8_t *>(comp_bfr.data());
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
  pq_chunked_state &state, bool return_filemetadata, const std::string &column_chunks_file_path)
{
  CompactProtocolWriter cpw(&buffer_);
  file_ender_s fendr;
  buffer_.resize(0);
  fendr.footer_len = static_cast<uint32_t>(cpw.write(state.md));
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
      for (auto &col : rowgroup.columns) { col.file_path = column_chunks_file_path; }
    }
    fendr.footer_len = static_cast<uint32_t>(cpw.write(state.md));
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
               parquet_writer_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mr))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::write(table_view const &table,
                                                    const table_metadata *metadata,
                                                    bool return_filemetadata,
                                                    const std::string column_chunks_file_path,
                                                    cudaStream_t stream)
{
  return _impl->write(table, metadata, return_filemetadata, column_chunks_file_path, stream);
}

// Forward to implementation
void writer::write_chunked_begin(pq_chunked_state &state)
{
  return _impl->write_chunked_begin(state);
}

// Forward to implementation
void writer::write_chunk(table_view const &table, pq_chunked_state &state)
{
  _impl->write_chunk(table, state);
}

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::write_chunked_end(
  pq_chunked_state &state, bool return_filemetadata, const std::string &column_chunks_file_path)
{
  return _impl->write_chunked_end(state, return_filemetadata, column_chunks_file_path);
}

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
  fendr.footer_len = static_cast<uint32_t>(cpw.write(md));
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
