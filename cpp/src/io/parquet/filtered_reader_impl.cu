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
 * @brief cuDF-IO Parquet reader class implementation
 */

 #include "reader_impl.hpp"

 #include <io/comp/gpuinflate.h>
 
 #include <cudf/table/table.hpp>
 #include <cudf/utilities/error.hpp>
 #include <cudf/utilities/traits.hpp>
 
 #include <rmm/thrust_rmm_allocator.h>
 #include <rmm/device_buffer.hpp>
 
 #include <algorithm>
 #include <array>
 #include <numeric>
 #include <regex>
 
 namespace cudf {
 namespace io {
 namespace detail {
 namespace parquet {
 // Import functionality that's independent of legacy code
 using namespace cudf::io::parquet;
 using namespace cudf::io;

 table_with_metadata reader::impl::read_filtered(size_type skip_rows,
                                                 size_type num_rows,
                                                 std::vector<std::vector<size_type>> const &row_group_list,
                                                 cudaStream_t stream)
 { 
   // Select only row groups required
   const auto selected_row_groups =
     _metadata->select_row_groups(row_group_list, skip_rows, num_rows);
 
   // Get a list of column data types
   std::vector<data_type> column_types;
   if (_metadata->get_num_row_groups() != 0) {
     for (const auto &col : _selected_columns) {
       auto const &col_schema =
         _metadata->get_schema(_metadata->get_row_group(0, 0).columns[col.first].schema_idx);
       auto const col_type = to_type_id(col_schema.type,
                                        col_schema.converted_type,
                                        _strings_to_categorical,
                                        _timestamp_type.id(),
                                        col_schema.decimal_scale);
       CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
       column_types.emplace_back(col_type);
     }
   }
 
   std::vector<std::unique_ptr<column>> out_columns;
   out_columns.reserve(column_types.size());
 
   if (selected_row_groups.size() != 0 && column_types.size() != 0) {
     // Descriptors for all the chunks that make up the selected columns
     const auto num_columns = _selected_columns.size();
     const auto num_chunks  = selected_row_groups.size() * num_columns;
     hostdevice_vector<gpu::ColumnChunkDesc> chunks(0, num_chunks, stream);
 
     // Association between each column chunk and its column
     std::vector<int> chunk_col_map(num_chunks);
     // Association between each column chunk and its source
     std::vector<size_type> chunk_source_map(num_chunks);
 
     // Tracker for eventually deallocating compressed and uncompressed data
     std::vector<rmm::device_buffer> page_data(num_chunks);
 
     // Keep track of column chunk file offsets
     std::vector<size_t> column_chunk_offsets(num_chunks);
 
     // Initialize column chunk information
     size_t total_decompressed_size = 0;
     auto remaining_rows            = num_rows;
     for (const auto &rg : selected_row_groups) {
       const auto &row_group       = _metadata->get_row_group(rg.index, rg.source_index);
       auto const row_group_start  = rg.start_row;
       auto const row_group_source = rg.source_index;
       auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
       auto const io_chunk_idx     = chunks.size();
 
       for (size_t i = 0; i < num_columns; ++i) {
         auto const col         = _selected_columns[i];
         auto const &col_meta   = row_group.columns[col.first].meta_data;
         auto const &col_schema = _metadata->get_schema(row_group.columns[col.first].schema_idx);
 
         // Spec requires each row group to contain exactly one chunk for every
         // column. If there are too many or too few, continue with best effort
         if (col.second != name_from_path(col_meta.path_in_schema)) {
           std::cerr << "Detected mismatched column chunk" << std::endl;
           continue;
         }
         if (chunks.size() >= chunks.max_size()) {
           std::cerr << "Detected too many column chunks" << std::endl;
           continue;
         }
 
         int32_t type_width;
         int32_t clock_rate;
         int8_t converted_type;
         std::tie(type_width, clock_rate, converted_type) =
           conversion_info(column_types[i].id(),
                           _timestamp_type.id(),
                           col_schema.type,
                           col_schema.converted_type,
                           col_schema.type_length);
 
         column_chunk_offsets[chunks.size()] =
           (col_meta.dictionary_page_offset != 0)
             ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
             : col_meta.data_page_offset;
 
         chunks.insert(gpu::ColumnChunkDesc(col_meta.total_compressed_size,
                                            nullptr,
                                            col_meta.num_values,
                                            col_schema.type,
                                            type_width,
                                            row_group_start,
                                            row_group_rows,
                                            col_schema.max_definition_level,
                                            col_schema.max_repetition_level,
                                            required_bits(col_schema.max_definition_level),
                                            required_bits(col_schema.max_repetition_level),
                                            col_meta.codec,
                                            converted_type,
                                            col_schema.decimal_scale,
                                            clock_rate));
 
         // Map each column chunk to its column index and its source index
         chunk_col_map[chunks.size() - 1]    = i;
         chunk_source_map[chunks.size() - 1] = row_group_source;
 
         if (col_meta.codec != Compression::UNCOMPRESSED) {
           total_decompressed_size += col_meta.total_uncompressed_size;
         }
       }
       // Read compressed chunk data to device memory
       read_column_chunks(page_data,
                          chunks,
                          io_chunk_idx,
                          chunks.size(),
                          column_chunk_offsets,
                          chunk_source_map,
                          stream);
 
       remaining_rows -= row_group.num_rows;
     }
     assert(remaining_rows <= 0);
 
     // Process dataset chunk pages into output columns
     const auto total_pages = count_page_headers(chunks, stream);
     if (total_pages > 0) {
       hostdevice_vector<gpu::PageInfo> pages(total_pages, total_pages, stream);
       rmm::device_buffer decomp_page_data;
 
       decode_page_headers(chunks, pages, stream);
       if (total_decompressed_size > 0) {
         decomp_page_data = decompress_page_data(chunks, pages, stream);
         // Free compressed data
         for (size_t c = 0; c < chunks.size(); c++) {
           if (chunks[c].codec != parquet::Compression::UNCOMPRESSED && page_data[c].size() != 0) {
             page_data[c].resize(0);
             page_data[c].shrink_to_fit();
           }
         }
       }
 
       std::vector<column_buffer> out_buffers;
       out_buffers.reserve(column_types.size());
       for (size_t i = 0; i < column_types.size(); ++i) {
         auto col                    = _selected_columns[i];
         auto const &first_row_group = _metadata->get_row_group(selected_row_groups[0].index,
                                                                selected_row_groups[0].source_index);
         auto &col_schema = _metadata->get_schema(first_row_group.columns[col.first].schema_idx);
         bool is_nullable = (col_schema.max_definition_level != 0);
         out_buffers.emplace_back(column_types[i], num_rows, is_nullable, stream, _mr);
       }
 
       decode_page_data(chunks, pages, skip_rows, num_rows, chunk_col_map, out_buffers, stream);
 
       for (size_t i = 0; i < column_types.size(); ++i) {
         out_columns.emplace_back(
           make_column(column_types[i], num_rows, out_buffers[i], stream, _mr));
       }
     }
   }
 
   // Create empty columns as needed
   for (size_t i = out_columns.size(); i < column_types.size(); ++i) {
     out_columns.emplace_back(make_empty_column(column_types[i]));
   }
 
   table_metadata out_metadata;
   // Return column names (must match order of returned columns)
   out_metadata.column_names.resize(_selected_columns.size());
   for (size_t i = 0; i < _selected_columns.size(); i++) {
     out_metadata.column_names[i] = _selected_columns[i].second;
   }
   // Return user metadata
   out_metadata.user_data = _metadata->get_key_value_metadata();
 
   return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
 }

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
