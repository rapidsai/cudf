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

#include <nvstrings/NVStrings.h>
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
constexpr parquet::Compression to_parquet_compression(
    compression_type compression) {
  switch (compression) {
    case compression_type::SNAPPY:
      return parquet::Compression::SNAPPY;
    case compression_type::NONE:
    default:
      return parquet::Compression::UNCOMPRESSED;
  }
}

}  // namespace

/**
 * @brief Helper class that adds parquet-specific column info
 **/
class parquet_column_view {
  using str_pair = std::pair<const char *, size_t>;

 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   **/
  explicit parquet_column_view(size_t id, column_view const &col, cudaStream_t stream)
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
    if (_string_type) {
      // FIXME: Use thrust to generate index without creating a NVStrings instance
      strings_column_view view{col};
      _nvstr =
          NVStrings::create_from_offsets(view.chars().data<char>(), view.size(),
                                         view.offsets().data<size_type>());

      _indexes = rmm::device_buffer(_data_count * sizeof(str_pair), stream);
      CUDF_EXPECTS(
          _nvstr->create_index(static_cast<str_pair *>(_indexes.data())) == 0,
          "Cannot retrieve string pairs");
      _data = _indexes.data();
    }
    _name = "_col" + std::to_string(_id);
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
  uint32_t *get_dict_data() { return _dict_data.data().get(); }
  uint32_t *get_dict_index() { return _dict_index.data().get(); }
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
  NVStrings *_nvstr = nullptr;
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
                                              uint32_t fragment_size, cudaStream_t stream)
{
  rmm::device_vector<statistics_group> frag_stats_group(num_fragments * num_columns);

  CUDA_TRY(gpu::InitFragmentStatistics(frag_stats_group.data().get(), frag.device_ptr(),
                                       col_desc.device_ptr(), num_fragments, num_columns,
                                       fragment_size, stream));
  CUDA_TRY(GatherColumnStatistics(frag_stats_chunk, frag_stats_group.data().get(),
                                  num_fragments * num_columns, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}



writer::impl::impl(std::string filepath, writer_options const &options,
                   rmm::mr::device_memory_resource *mr)
    : _mr(mr) {
  compression_kind_ = to_parquet_compression(options.compression);
  stats_granularity_ = options.stats_granularity;

  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::impl::write(table_view const &table, cudaStream_t stream) {
  size_type num_columns = table.num_columns();
  size_type num_rows = 0;

  // Wrapper around cudf columns to attach parquet-specific type info
  std::vector<parquet_column_view> parquet_columns;
  for (auto it = table.begin(); it < table.end(); ++it) {
    const auto col = *it;
    const auto current_id = parquet_columns.size();

    num_rows = std::max<uint32_t>(num_rows, col.size());
    parquet_columns.emplace_back(current_id, col, stream);
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
  uint32_t num_chunks = num_rowgroups * num_columns;
  hostdevice_vector<gpu::EncColumnChunk> chunks(num_columns * num_fragments);
  rmm::device_vector<statistics_chunk> frag_stats_chunk;
  if (stats_granularity_ > statistics_freq::statistics_none) {
    frag_stats_chunk.resize(num_fragments * num_columns);
    gather_fragment_statistics(frag_stats_chunk.data().get(), fragments, col_desc,
                               num_columns, num_fragments, fragment_size, stream);
  }

  // Write file header
  file_header_s fhdr;
  fhdr.magic = PARQUET_MAGIC;
  outfile_.write(reinterpret_cast<char *>(&fhdr), sizeof(fhdr));

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
void writer::write_all(table_view const &table, cudaStream_t stream) {
  _impl->write(table, stream);
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
