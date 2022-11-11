/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
 * @file column_buffer.cpp
 * @brief cuDF-IO column_buffer class implementation
 */

#include "column_buffer.hpp"
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

namespace cudf {
namespace io {
namespace detail {

void column_buffer::create(size_type _size,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  size = _size;

  switch (type.id()) {
    case type_id::STRING:
      _strings = std::make_unique<rmm::device_uvector<string_index_pair>>(
        cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(size, stream));
      break;

    // list columns store a buffer of int32's as offsets to represent
    // their individual rows
    case type_id::LIST: _data = create_data(data_type{type_id::INT32}, size, stream, mr); break;

    // struct columns store no data themselves.  just validity and children.
    case type_id::STRUCT: break;

    default: _data = create_data(type, size, stream, mr); break;
  }
  if (is_nullable) {
    _null_mask =
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, rmm::cuda_stream_view(stream), mr);
  }
}

/**
 * @copydoc cudf::io::detail::make_column
 */
std::unique_ptr<column> make_column(column_buffer& buffer,
                                    column_name_info* schema_info,
                                    std::optional<reader_column_schema> const& schema,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  if (schema_info != nullptr) { schema_info->name = buffer.name; }

  switch (buffer.type.id()) {
    case type_id::STRING:
      if (schema.value_or(reader_column_schema{}).is_enabled_convert_binary_to_strings()) {
        if (schema_info != nullptr) {
          schema_info->children.push_back(column_name_info{"offsets"});
          schema_info->children.push_back(column_name_info{"chars"});
        }

        return make_strings_column(*buffer._strings, stream, mr);
      } else {
        // convert to binary
        auto const string_col = make_strings_column(*buffer._strings, stream, mr);
        auto const num_rows   = string_col->size();
        auto col_content      = string_col->release();

        // convert to uint8 column, strings are currently stores as int8
        auto contents =
          col_content.children[strings_column_view::chars_column_index].release()->release();
        auto data      = contents.data.release();
        auto null_mask = contents.null_mask.release();

        auto uint8_col = std::make_unique<column>(data_type{type_id::UINT8},
                                                  data->size(),
                                                  std::move(*data),
                                                  std::move(*null_mask),
                                                  UNKNOWN_NULL_COUNT);

        if (schema_info != nullptr) {
          schema_info->children.push_back(column_name_info{"offsets"});
          schema_info->children.push_back(column_name_info{"binary"});
        }

        return make_lists_column(
          num_rows,
          std::move(col_content.children[strings_column_view::offsets_column_index]),
          std::move(uint8_col),
          UNKNOWN_NULL_COUNT,
          std::move(*col_content.null_mask));
      }

    case type_id::LIST: {
      // make offsets column
      auto offsets =
        std::make_unique<column>(data_type{type_id::INT32}, buffer.size, std::move(buffer._data));

      column_name_info* child_info = nullptr;
      if (schema_info != nullptr) {
        schema_info->children.push_back(column_name_info{"offsets"});
        schema_info->children.push_back(column_name_info{""});
        child_info = &schema_info->children.back();
      }

      CUDF_EXPECTS(not schema.has_value() or schema->get_num_children() > 0,
                   "Invalid schema provided for read, expected child data for list!");
      auto const child_schema = schema.has_value()
                                  ? std::make_optional<reader_column_schema>(schema->child(0))
                                  : std::nullopt;

      // make child column
      CUDF_EXPECTS(buffer.children.size() > 0, "Encountered malformed column_buffer");
      auto child = make_column(buffer.children[0], child_info, child_schema, stream, mr);

      // make the final list column (note : size is the # of offsets, so our actual # of rows is 1
      // less)
      return make_lists_column(buffer.size - 1,
                               std::move(offsets),
                               std::move(child),
                               buffer._null_count,
                               std::move(buffer._null_mask),
                               stream,
                               mr);
    } break;

    case type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> output_children;
      output_children.reserve(buffer.children.size());
      for (size_t i = 0; i < buffer.children.size(); ++i) {
        column_name_info* child_info = nullptr;
        if (schema_info != nullptr) {
          schema_info->children.push_back(column_name_info{""});
          child_info = &schema_info->children.back();
        }

        CUDF_EXPECTS(not schema.has_value() or schema->get_num_children() > i,
                     "Invalid schema provided for read, expected more child data for struct!");
        auto const child_schema = schema.has_value()
                                    ? std::make_optional<reader_column_schema>(schema->child(i))
                                    : std::nullopt;

        output_children.emplace_back(
          make_column(buffer.children[i], child_info, child_schema, stream, mr));
      }

      return make_structs_column(buffer.size,
                                 std::move(output_children),
                                 buffer._null_count,
                                 std::move(buffer._null_mask),
                                 stream,
                                 mr);
    } break;

    default: {
      return std::make_unique<column>(buffer.type,
                                      buffer.size,
                                      std::move(buffer._data),
                                      std::move(buffer._null_mask),
                                      buffer._null_count);
    }
  }
}

/**
 * @copydoc cudf::io::detail::empty_like
 */
std::unique_ptr<column> empty_like(column_buffer& buffer,
                                   column_name_info* schema_info,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (schema_info != nullptr) { schema_info->name = buffer.name; }

  switch (buffer.type.id()) {
    case type_id::LIST: {
      // make offsets column
      auto offsets = cudf::make_empty_column(type_id::INT32);

      column_name_info* child_info = nullptr;
      if (schema_info != nullptr) {
        schema_info->children.push_back(column_name_info{"offsets"});
        schema_info->children.push_back(column_name_info{""});
        child_info = &schema_info->children.back();
      }

      // make child column
      CUDF_EXPECTS(buffer.children.size() > 0, "Encountered malformed column_buffer");
      auto child = empty_like(buffer.children[0], child_info, stream, mr);

      // make the final list column
      return make_lists_column(0,
                               std::move(offsets),
                               std::move(child),
                               buffer._null_count,
                               std::move(buffer._null_mask),
                               stream,
                               mr);
    } break;

    case type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> output_children;
      output_children.reserve(buffer.children.size());
      std::transform(buffer.children.begin(),
                     buffer.children.end(),
                     std::back_inserter(output_children),
                     [&](column_buffer& col) {
                       column_name_info* child_info = nullptr;
                       if (schema_info != nullptr) {
                         schema_info->children.push_back(column_name_info{""});
                         child_info = &schema_info->children.back();
                       }
                       return empty_like(col, child_info, stream, mr);
                     });

      return make_structs_column(0,
                                 std::move(output_children),
                                 buffer._null_count,
                                 std::move(buffer._null_mask),
                                 stream,
                                 mr);
    } break;

    default: return cudf::make_empty_column(buffer.type);
  }
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
