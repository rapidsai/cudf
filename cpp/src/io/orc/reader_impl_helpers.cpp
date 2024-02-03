/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "reader_impl_helpers.hpp"

namespace cudf::io::orc::detail {


std::size_t gather_stream_info_and_column_desc(
  std::size_t stripe_index,
  std::size_t level,
  orc::StripeInformation const* stripeinfo,
  orc::StripeFooter const* stripefooter,
  host_span<int const> orc2gdf,
  host_span<orc::SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  std::size_t* num_dictionary_entries,
  std::size_t* stream_idx,
  std::optional<std::vector<orc_stream_info>*> const& stream_info,
  std::optional<cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>*> const& chunks)
{
  CUDF_EXPECTS(stream_info.has_value() ^ chunks.has_value(),
    "Either stream_info or chunks must be provided, but not both.");

  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;

  auto const get_stream_index_type = [](orc::StreamKind kind) {
    switch (kind) {
      case orc::DATA: return gpu::CI_DATA;
      case orc::LENGTH:
      case orc::SECONDARY: return gpu::CI_DATA2;
      case orc::DICTIONARY_DATA: return gpu::CI_DICTIONARY;
      case orc::PRESENT: return gpu::CI_PRESENT;
      case orc::ROW_INDEX: return gpu::CI_INDEX;
      default:
        // Skip this stream as it's not strictly required
        return gpu::CI_NUM_STREAMS;
    }
  };

  for (auto const& stream : stripefooter->streams) {
    if (!stream.column_id || *stream.column_id >= orc2gdf.size()) {
      // TODO: fix dst to src
      src_offset += stream.length;
      continue;
    }

    auto const column_id = *stream.column_id;
    auto col             = orc2gdf[column_id];

    if (col == -1 and apply_struct_map) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      auto const schema_type = types[column_id];
      if (! schema_type.subtypes.empty() && schema_type.kind == orc::STRUCT &&
          stream.kind == orc::PRESENT) {
            
          for (auto const& idx : schema_type.subtypes) {
            auto const child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
            if (child_idx >= 0) {
              col                             = child_idx;
              if(chunks.has_value()) {
                auto& chunk                     = (*chunks.value())[stripe_index][col];
                chunk.strm_id[gpu::CI_PRESENT]  = *stream_idx;
                chunk.strm_len[gpu::CI_PRESENT] = stream.length;
              }
            }
          }
        }
    }
    if (col != -1) {
      if (chunks.has_value()) {
         if (src_offset >= stripeinfo->indexLength || use_index) {
        auto const index_type = get_stream_index_type(stream.kind);
        if (index_type < gpu::CI_NUM_STREAMS) {
          auto& chunk           = (*chunks.value())[stripe_index][col];
          chunk.strm_id[index_type]  = *stream_idx;
          chunk.strm_len[index_type] = stream.length;
          // NOTE: skip_count field is temporarily used to track the presence of index streams
          chunk.skip_count |= 1 << index_type;

          if (index_type == gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
            *num_dictionary_entries += stripefooter->columns[column_id].dictionarySize;
          }
        }
      }
      (*stream_idx)++;
      } else { // not chunks.has_value()
        stream_info.value().emplace_back(stripeinfo->offset + src_offset,
                               dst_offset,
                               stream.length,
                               stream_id_info{stripe_index,
                               level,
                               column_id,
                               stream.kind});
      }

      
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

std::unique_ptr<column> create_empty_column(size_type orc_col_id,
                                            aggregate_orc_metadata const& metadata,
                                            host_span<std::string const> decimal128_columns,
                                            bool use_np_dtypes,
                                            data_type timestamp_type,
                                            column_name_info& schema_info,
                                            rmm::cuda_stream_view stream)
{
  schema_info.name = metadata.column_name(0, orc_col_id);
  auto const kind  = metadata.get_col_type(orc_col_id).kind;
  auto const type  = to_cudf_type(kind,
                                 use_np_dtypes,
                                 timestamp_type.id(),
                                 to_cudf_decimal_type(decimal128_columns, metadata, orc_col_id));

  switch (kind) {
    case orc::LIST: {
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("");
      return make_lists_column(0,
                               make_empty_column(type_id::INT32),
                               create_empty_column(metadata.get_col_type(orc_col_id).subtypes[0],
                                                   metadata,
                                                   decimal128_columns,
                                                   use_np_dtypes,
                                                   timestamp_type,
                                                   schema_info.children.back(),
                                                   stream),
                               0,
                               rmm::device_buffer{0, stream},
                               stream);
    }
    case orc::MAP: {
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("struct");
      auto const child_column_ids = metadata.get_col_type(orc_col_id).subtypes;
      auto& children_schema       = schema_info.children.back().children;
      std::vector<std::unique_ptr<column>> child_columns;
      for (std::size_t idx = 0; idx < metadata.get_col_type(orc_col_id).subtypes.size(); idx++) {
        children_schema.emplace_back("");
        child_columns.push_back(create_empty_column(child_column_ids[idx],
                                                    metadata,
                                                    decimal128_columns,
                                                    use_np_dtypes,
                                                    timestamp_type,
                                                    schema_info.children.back().children.back(),
                                                    stream));
        children_schema[idx].name = get_map_child_col_name(idx);
      }
      return make_lists_column(
        0,
        make_empty_column(type_id::INT32),
        make_structs_column(0, std::move(child_columns), 0, rmm::device_buffer{0, stream}, stream),
        0,
        rmm::device_buffer{0, stream},
        stream);
    }

    case orc::STRUCT: {
      std::vector<std::unique_ptr<column>> child_columns;
      for (auto const col : metadata.get_col_type(orc_col_id).subtypes) {
        schema_info.children.emplace_back("");
        child_columns.push_back(create_empty_column(col,
                                                    metadata,
                                                    decimal128_columns,
                                                    use_np_dtypes,
                                                    timestamp_type,
                                                    schema_info.children.back(),
                                                    stream));
      }
      return make_structs_column(
        0, std::move(child_columns), 0, rmm::device_buffer{0, stream}, stream);
    }

    case orc::DECIMAL: {
      int32_t scale = 0;
      if (type == type_id::DECIMAL32 or type == type_id::DECIMAL64 or type == type_id::DECIMAL128) {
        scale = -static_cast<int32_t>(metadata.get_types()[orc_col_id].scale.value_or(0));
      }
      return make_empty_column(data_type(type, scale));
    }

    default: return make_empty_column(type);
  }
}

column_buffer assemble_buffer(size_type orc_col_id,
                              std::size_t level,
                              reader_column_meta const& col_meta,
                              aggregate_orc_metadata const& metadata,
                              column_hierarchy const& selected_columns,
                              std::vector<std::vector<column_buffer>>& col_buffers,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  auto const col_id = col_meta.orc_col_map[level][orc_col_id];
  auto& col_buffer  = col_buffers[level][col_id];

  col_buffer.name = metadata.column_name(0, orc_col_id);
  auto kind       = metadata.get_col_type(orc_col_id).kind;
  switch (kind) {
    case orc::LIST:
    case orc::STRUCT: {
      auto const& children_indices = selected_columns.children.at(orc_col_id);
      for (auto const child_id : children_indices) {
        col_buffer.children.emplace_back(assemble_buffer(
          child_id, level + 1, col_meta, metadata, selected_columns, col_buffers, stream, mr));
      }
    } break;

    case orc::MAP: {
      std::vector<column_buffer> child_col_buffers;
      // Get child buffers
      auto const& children_indices = selected_columns.children.at(orc_col_id);
      for (std::size_t idx = 0; idx < children_indices.size(); idx++) {
        auto const col = children_indices[idx];
        child_col_buffers.emplace_back(assemble_buffer(
          col, level + 1, col_meta, metadata, selected_columns, col_buffers, stream, mr));
        child_col_buffers.back().name = get_map_child_col_name(idx);
      }
      // Create a struct buffer
      auto num_rows = child_col_buffers[0].size;
      auto struct_buffer =
        column_buffer(cudf::data_type(type_id::STRUCT), num_rows, false, stream, mr);
      struct_buffer.children = std::move(child_col_buffers);
      struct_buffer.name     = "struct";

      col_buffer.children.emplace_back(std::move(struct_buffer));
    } break;

    default: break;
  }

  return std::move(col_buffer);
}

}  // namespace cudf::io::orc::detail
