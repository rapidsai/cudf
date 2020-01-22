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
 * @file reader_impl.cu
 * @brief cuDF-IO Avro reader class implementation
 **/

#include "reader_impl.hpp"

#include <io/comp/gpuinflate.h>

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace avro {

// Import functionality that's independent of legacy code
using namespace cudf::io::avro;
using namespace cudf::io;

namespace {

/**
 * @brief Function that translates Avro data kind to cuDF type enum
 **/
type_id to_type_id(const avro::schema_entry *col) {
  switch (col->kind) {
    case avro::type_boolean:
      return type_id::BOOL8;
    case avro::type_int:
      return type_id::INT32;
    case avro::type_long:
      return type_id::INT64;
    case avro::type_float:
      return type_id::FLOAT32;
    case avro::type_double:
      return type_id::FLOAT64;
    case avro::type_bytes:
    case avro::type_string:
      return type_id::STRING;
    case avro::type_enum:
      return (!col->symbols.empty()) ? type_id::STRING : type_id::INT32;
    default:
      return type_id::EMPTY;
  }
}

}  // namespace

/**
 * @brief A helper wrapper for Avro file metadata. Provides some additional
 * convenience methods for initializing and accessing the metadata and schema
 **/
class metadata : public file_metadata {
 public:
  explicit metadata(datasource *const src) : source(src) {}

  /**
   * @brief Initializes the parser and filters down to a subset of rows
   *
   * @param[in,out] row_start Starting row of the selection
   * @param[in,out] row_count Total number of rows selected
   **/
  void init_and_select_rows(int &row_start, int &row_count) {
    const auto buffer = source->get_buffer(0, source->size());
    avro::container pod(buffer->data(), buffer->size());
    CUDF_EXPECTS(pod.parse(this, row_count, row_start),
                 "Cannot parse metadata");
    row_start = skip_rows;
    row_count = num_rows;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param[in] use_names List of column names to select
   *
   * @return List of column names
   **/
  auto select_columns(std::vector<std::string> use_names) {
    std::vector<std::pair<int, std::string>> selection;

    const auto num_avro_columns = static_cast<int>(columns.size());
    if (!use_names.empty()) {
      int index = 0;
      for (const auto &use_name : use_names) {
        for (int i = 0; i < num_avro_columns; ++i, ++index) {
          if (index >= num_avro_columns) {
            index = 0;
          }
          if (columns[index].name == use_name &&
              type_id::EMPTY !=
                  to_type_id(&schema[columns[index].schema_data_idx])) {
            selection.emplace_back(index, columns[index].name);
            index++;
            break;
          }
        }
      }
    } else {
      for (int i = 0; i < num_avro_columns; ++i) {
        auto col_type = to_type_id(&schema[columns[i].schema_data_idx]);
        CUDF_EXPECTS(col_type != type_id::EMPTY, "Unsupported data type");
        selection.emplace_back(i, columns[i].name);
      }
    }
    CUDF_EXPECTS(selection.size() > 0, "Filtered out all columns");

    return selection;
  }

 private:
  datasource *const source;
};

rmm::device_buffer reader::impl::decompress_data(
    const rmm::device_buffer &comp_block_data, cudaStream_t stream) {
  size_t uncompressed_data_size = 0;
  hostdevice_vector<gpu_inflate_input_s> inflate_in(
      _metadata->block_list.size());
  hostdevice_vector<gpu_inflate_status_s> inflate_out(
      _metadata->block_list.size());

  if (_metadata->codec == "deflate") {
    // Guess an initial maximum uncompressed block size
    uint32_t initial_blk_len = (_metadata->max_block_size * 2 + 0xfff) & ~0xfff;
    uncompressed_data_size = initial_blk_len * _metadata->block_list.size();
    for (size_t i = 0; i < inflate_in.size(); ++i) {
      inflate_in[i].dstSize = initial_blk_len;
    }
  } else if (_metadata->codec == "snappy") {
    // Extract the uncompressed length from the snappy stream
    for (size_t i = 0; i < _metadata->block_list.size(); i++) {
      const auto buffer =
          _source->get_buffer(_metadata->block_list[i].offset, 4);
      const uint8_t *blk = buffer->data();
      uint32_t blk_len = blk[0];
      if (blk_len > 0x7f) {
        blk_len = (blk_len & 0x7f) | (blk[1] << 7);
        if (blk_len > 0x3fff) {
          blk_len = (blk_len & 0x3fff) | (blk[2] << 14);
          if (blk_len > 0x1fffff) {
            blk_len = (blk_len & 0x1fffff) | (blk[3] << 21);
          }
        }
      }
      inflate_in[i].dstSize = blk_len;
      uncompressed_data_size += blk_len;
    }
  } else {
    CUDF_FAIL("Unsupported compression codec\n");
  }

  rmm::device_buffer decomp_block_data(uncompressed_data_size, stream);

  const auto base_offset = _metadata->block_list[0].offset;
  for (size_t i = 0, dst_pos = 0; i < _metadata->block_list.size(); i++) {
    const auto src_pos = _metadata->block_list[i].offset - base_offset;

    inflate_in[i].srcDevice =
        static_cast<const uint8_t *>(comp_block_data.data()) + src_pos;
    inflate_in[i].srcSize = _metadata->block_list[i].size;
    inflate_in[i].dstDevice =
        static_cast<uint8_t *>(decomp_block_data.data()) + dst_pos;

    // Update blocks offsets & sizes to refer to uncompressed data
    _metadata->block_list[i].offset = dst_pos;
    _metadata->block_list[i].size =
        static_cast<uint32_t>(inflate_in[i].dstSize);
    dst_pos += _metadata->block_list[i].size;
  }

  for (int loop_cnt = 0; loop_cnt < 2; loop_cnt++) {
    CUDA_TRY(cudaMemcpyAsync(inflate_in.device_ptr(), inflate_in.host_ptr(),
                             inflate_in.memory_size(), cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(cudaMemsetAsync(inflate_out.device_ptr(), 0,
                             inflate_out.memory_size(), stream));
    if (_metadata->codec == "deflate") {
      CUDA_TRY(gpuinflate(inflate_in.device_ptr(), inflate_out.device_ptr(),
                          inflate_in.size(), 0, stream));
    } else if (_metadata->codec == "snappy") {
      CUDA_TRY(gpu_unsnap(inflate_in.device_ptr(), inflate_out.device_ptr(),
                          inflate_in.size(), stream));
    } else {
      CUDF_FAIL("Unsupported compression codec\n");
    }
    CUDA_TRY(cudaMemcpyAsync(inflate_out.host_ptr(), inflate_out.device_ptr(),
                             inflate_out.memory_size(), cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    // Check if larger output is required, as it's not known ahead of time
    if (_metadata->codec == "deflate" && !loop_cnt) {
      size_t actual_uncompressed_size = 0;
      for (size_t i = 0; i < _metadata->block_list.size(); i++) {
        // If error status is 1 (buffer too small), the `bytes_written` field
        // is actually contains the uncompressed data size
        if (inflate_out[i].status == 1 &&
            inflate_out[i].bytes_written > inflate_in[i].dstSize) {
          inflate_in[i].dstSize = inflate_out[i].bytes_written;
        }
        actual_uncompressed_size += inflate_in[i].dstSize;
      }
      if (actual_uncompressed_size > uncompressed_data_size) {
        decomp_block_data.resize(actual_uncompressed_size);
        for (size_t i = 0, dst_pos = 0; i < _metadata->block_list.size(); i++) {
          auto dst_base = static_cast<uint8_t *>(decomp_block_data.data());
          inflate_in[i].dstDevice = dst_base + dst_pos;

          _metadata->block_list[i].offset = dst_pos;
          _metadata->block_list[i].size =
              static_cast<uint32_t>(inflate_in[i].dstSize);
          dst_pos += _metadata->block_list[i].size;
        }
      } else {
        break;
      }
    } else {
      break;
    }
  }

  return decomp_block_data;
}

void reader::impl::decode_data(
    const rmm::device_buffer &block_data,
    const std::vector<std::pair<uint32_t, uint32_t>> &dict,
    const hostdevice_vector<uint8_t> &global_dictionary,
    size_t total_dictionary_entries, size_t num_rows,
    std::vector<std::pair<int, std::string>> selection,
    std::vector<column_buffer> &out_buffers, cudaStream_t stream) {
  // Build gpu schema
  hostdevice_vector<gpu::schemadesc_s> schema_desc(_metadata->schema.size());
  uint32_t min_row_data_size = 0;
  int skip_field_cnt = 0;
  for (size_t i = 0; i < _metadata->schema.size(); i++) {
    type_kind_e kind = _metadata->schema[i].kind;
    if (skip_field_cnt != 0) {
      // Exclude union members from min_row_data_size
      skip_field_cnt += _metadata->schema[i].num_children - 1;
    } else {
      switch (kind) {
        case type_union:
          skip_field_cnt = _metadata->schema[i].num_children;
          // fall through
        case type_boolean:
        case type_int:
        case type_long:
        case type_bytes:
        case type_string:
        case type_enum:
          min_row_data_size += 1;
          break;
        case type_float:
          min_row_data_size += 4;
          break;
        case type_double:
          min_row_data_size += 8;
          break;
        default:
          break;
      }
    }
    if (kind == type_enum && !_metadata->schema[i].symbols.size()) {
      kind = type_int;
    }
    schema_desc[i].kind = kind;
    schema_desc[i].count =
        (kind == type_enum) ? 0 : (uint32_t)_metadata->schema[i].num_children;
    schema_desc[i].dataptr = nullptr;
    CUDF_EXPECTS(kind != type_union || _metadata->schema[i].num_children < 2 ||
                     (_metadata->schema[i].num_children == 2 &&
                      (_metadata->schema[i + 1].kind == type_null ||
                       _metadata->schema[i + 2].kind == type_null)),
                 "Union with non-null type not currently supported");
  }
  std::vector<void *> valid_alias(out_buffers.size(), nullptr);
  for (size_t i = 0; i < out_buffers.size(); i++) {
    const auto col_idx = selection[i].first;
    int schema_data_idx = _metadata->columns[col_idx].schema_data_idx;
    int schema_null_idx = _metadata->columns[col_idx].schema_null_idx;

    schema_desc[schema_data_idx].dataptr = out_buffers[i].data();
    if (schema_null_idx >= 0) {
      if (!schema_desc[schema_null_idx].dataptr) {
        schema_desc[schema_null_idx].dataptr = out_buffers[i].null_mask();
      } else {
        valid_alias[i] = schema_desc[schema_null_idx].dataptr;
      }
    }
    if (_metadata->schema[schema_data_idx].kind == type_enum) {
      schema_desc[schema_data_idx].count = dict[i].first;
    }
    CUDA_TRY(cudaMemsetAsync(out_buffers[i].null_mask(), -1,
                             bitmask_allocation_size_bytes(num_rows), stream));
  }
  rmm::device_buffer block_list(
      _metadata->block_list.data(),
      _metadata->block_list.size() * sizeof(block_desc_s), stream);
  CUDA_TRY(cudaMemcpyAsync(schema_desc.device_ptr(), schema_desc.host_ptr(),
                           schema_desc.memory_size(), cudaMemcpyHostToDevice,
                           stream));

  CUDA_TRY(gpu::DecodeAvroColumnData(
      static_cast<block_desc_s *>(block_list.data()), schema_desc.device_ptr(),
      reinterpret_cast<gpu::nvstrdesc_s *>(global_dictionary.device_ptr()),
      static_cast<const uint8_t *>(block_data.data()),
      static_cast<uint32_t>(_metadata->block_list.size()),
      static_cast<uint32_t>(schema_desc.size()),
      static_cast<uint32_t>(total_dictionary_entries), _metadata->num_rows,
      _metadata->skip_rows, min_row_data_size, stream));

  // Copy valid bits that are shared between columns
  for (size_t i = 0; i < out_buffers.size(); i++) {
    if (valid_alias[i] != nullptr) {
      CUDA_TRY(cudaMemcpyAsync(out_buffers[i].null_mask(), valid_alias[i],
                               out_buffers[i].null_mask_size(),
                               cudaMemcpyHostToDevice, stream));
    }
  }
  CUDA_TRY(cudaMemcpyAsync(schema_desc.host_ptr(), schema_desc.device_ptr(),
                           schema_desc.memory_size(), cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < out_buffers.size(); i++) {
    const auto col_idx = selection[i].first;
    const auto schema_null_idx = _metadata->columns[col_idx].schema_null_idx;
    out_buffers[i].null_count() =
        (schema_null_idx >= 0) ? schema_desc[schema_null_idx].count : 0;
  }
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
    : _source(std::move(source)), _mr(mr), _columns(options.columns) {
  // Open the source Avro dataset metadata
  _metadata = std::make_unique<metadata>(_source.get());
}

table_with_metadata reader::impl::read(int skip_rows, int num_rows,
                                       cudaStream_t stream) {
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata metadata_out;

  // Select and read partial metadata / schema within the subset of rows
  _metadata->init_and_select_rows(skip_rows, num_rows);

  // Select only columns required by the options
  auto selected_columns = _metadata->select_columns(_columns);
  if (selected_columns.size() != 0) {
    // Get a list of column data types
    std::vector<data_type> column_types;
    for (const auto &col : selected_columns) {
      auto &col_schema =
          _metadata->schema[_metadata->columns[col.first].schema_data_idx];

      auto col_type = to_type_id(&col_schema);
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
      column_types.emplace_back(col_type);
    }

    if (_metadata->total_data_size > 0) {
      const auto buffer = _source->get_buffer(_metadata->block_list[0].offset,
                                              _metadata->total_data_size);
      rmm::device_buffer block_data(buffer->data(), buffer->size(), stream);

      if (_metadata->codec != "" && _metadata->codec != "null") {
        auto decomp_block_data = decompress_data(block_data, stream);
        block_data = std::move(decomp_block_data);
      } else {
        auto dst_ofs = _metadata->block_list[0].offset;
        for (size_t i = 0; i < _metadata->block_list.size(); i++) {
          _metadata->block_list[i].offset -= dst_ofs;
        }
      }

      size_t total_dictionary_entries = 0;
      size_t dictionary_data_size = 0;
      std::vector<std::pair<uint32_t, uint32_t>> dict(column_types.size());
      for (size_t i = 0; i < column_types.size(); ++i) {
        auto col_idx = selected_columns[i].first;
        auto &col_schema =
            _metadata->schema[_metadata->columns[col_idx].schema_data_idx];
        dict[i].first = static_cast<uint32_t>(total_dictionary_entries);
        dict[i].second = static_cast<uint32_t>(col_schema.symbols.size());
        total_dictionary_entries += dict[i].second;
        for (const auto &sym : col_schema.symbols) {
          dictionary_data_size += sym.length();
        }
      }

      hostdevice_vector<uint8_t> global_dictionary(
          total_dictionary_entries * sizeof(gpu::nvstrdesc_s) +
          dictionary_data_size);
      if (total_dictionary_entries > 0) {
        size_t dict_pos = total_dictionary_entries * sizeof(gpu::nvstrdesc_s);
        for (size_t i = 0; i < column_types.size(); ++i) {
          auto col_idx = selected_columns[i].first;
          auto &col_schema =
              _metadata->schema[_metadata->columns[col_idx].schema_data_idx];
          auto index = &(reinterpret_cast<gpu::nvstrdesc_s *>(
              global_dictionary.host_ptr()))[dict[i].first];
          for (size_t j = 0; j < dict[i].second; j++) {
            size_t len = col_schema.symbols[j].length();
            char *ptr = reinterpret_cast<char *>(
                global_dictionary.device_ptr() + dict_pos);
            index[j].ptr = ptr;
            index[j].count = len;
            memcpy(global_dictionary.host_ptr() + dict_pos,
                   col_schema.symbols[j].c_str(), len);
            dict_pos += len;
          }
        }
        CUDA_TRY(cudaMemcpyAsync(
            global_dictionary.device_ptr(), global_dictionary.host_ptr(),
            global_dictionary.memory_size(), cudaMemcpyHostToDevice, stream));
      }

      std::vector<column_buffer> out_buffers;
      for (size_t i = 0; i < column_types.size(); ++i) {
        out_buffers.emplace_back(column_types[i], num_rows, stream, _mr);
      }

      decode_data(block_data, dict, global_dictionary, total_dictionary_entries,
                  num_rows, selected_columns, out_buffers, stream);

      for (size_t i = 0; i < column_types.size(); ++i) {
        out_columns.emplace_back(make_column(column_types[i], num_rows,
                                             out_buffers[i], stream, _mr));
      }
    }
  }

  // Return column names (must match order of returned columns)
  metadata_out.column_names.resize(selected_columns.size());
  for (size_t i = 0; i < selected_columns.size(); i++) {
    metadata_out.column_names[i] = selected_columns[i].second;
  }
  // Return user metadata
  metadata_out.user_data = _metadata->user_data;

  return { std::make_unique<table>(std::move(out_columns)), std::move(metadata_out) };
}

// Forward to implementation
reader::reader(std::string filepath, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(filepath), options, mr)) {
}

// Forward to implementation
reader::reader(const char *buffer, size_t length, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(buffer, length), options,
                                   mr)) {}

// Forward to implementation
reader::reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(file), options, mr)) {}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream) {
  return _impl->read(0, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type skip_rows,
                                         size_type num_rows,
                                         cudaStream_t stream) {
  return _impl->read(skip_rows, (num_rows != 0) ? num_rows : -1, stream);
}

}  // namespace avro
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
