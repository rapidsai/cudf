/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf::io::detail {

void gather_column_buffer::allocate_strings_data(rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(type.id() == type_id::STRING, "allocate_strings_data called for non-string column");
  // The contents of _strings will never be directly returned to the user.
  // Due to the fact that make_strings_column copies the input data to
  // produce its outputs, _strings is actually a temporary. As a result, we
  // do not pass the provided mr to the call to
  // make_zeroed_device_uvector_async here and instead let it use the
  // default rmm memory resource.
  _strings = std::make_unique<rmm::device_uvector<string_index_pair>>(
    cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
      size, stream, rmm::mr::get_current_device_resource()));
}

std::unique_ptr<column> gather_column_buffer::make_string_column_impl(rmm::cuda_stream_view stream)
{
  // make_strings_column allocates new memory, it does not simply move
  // from the inputs, so we need to pass it the memory resource given to
  // the buffer on construction so that the memory is allocated using the
  // resource that the calling code expected.
  return make_strings_column(*_strings, stream, _mr);
}

void inline_column_buffer::allocate_strings_data(rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(type.id() == type_id::STRING, "allocate_strings_data called for non-string column");
  // size + 1 for final offset. _string_data will be initialized later.
  _data = create_data(data_type{type_id::INT32}, size + 1, stream, _mr);
}

void inline_column_buffer::create_string_data(size_t num_bytes, rmm::cuda_stream_view stream)
{
  _string_data = rmm::device_buffer(num_bytes, stream, _mr);
}

std::unique_ptr<column> inline_column_buffer::make_string_column_impl(rmm::cuda_stream_view stream)
{
  // no need for copies, just transfer ownership of the data_buffers to the columns
  auto const state = mask_state::UNALLOCATED;
  auto str_col =
    _string_data.size() == 0
      ? make_empty_column(data_type{type_id::INT8})
      : std::make_unique<column>(data_type{type_id::INT8},
                                 string_size(),
                                 std::move(_string_data),
                                 cudf::detail::create_null_mask(size, state, stream, _mr),
                                 state_null_count(state, size),
                                 std::vector<std::unique_ptr<column>>{});
  auto offsets_col =
    std::make_unique<column>(data_type{type_to_id<size_type>()},
                             size + 1,
                             std::move(_data),
                             cudf::detail::create_null_mask(size + 1, state, stream, _mr),
                             state_null_count(state, size + 1),
                             std::vector<std::unique_ptr<column>>{});

  return make_strings_column(
    size, std::move(offsets_col), std::move(str_col), null_count(), std::move(_null_mask));
}

namespace {

/**
 * @brief Recursively copy `name` and `user_data` fields of one buffer to another.
 *
 * @param buff The old output buffer
 * @param new_buff The new output buffer
 */
template <class string_policy>
void copy_buffer_data(string_policy const& buff, string_policy& new_buff)
{
  new_buff.name      = buff.name;
  new_buff.user_data = buff.user_data;
  for (auto const& child : buff.children) {
    auto& new_child = new_buff.children.emplace_back(string_policy(child.type, child.is_nullable));
    copy_buffer_data(child, new_child);
  }
}

}  // namespace

template <class string_policy>
void column_buffer_base<string_policy>::create(size_type _size,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  size = _size;
  _mr  = mr;

  switch (type.id()) {
    case type_id::STRING: static_cast<string_policy*>(this)->allocate_strings_data(stream); break;

    // list columns store a buffer of int32's as offsets to represent
    // their individual rows
    case type_id::LIST: _data = create_data(data_type{type_id::INT32}, size, stream, _mr); break;

    // struct columns store no data themselves.  just validity and children.
    case type_id::STRUCT: break;

    default: _data = create_data(type, size, stream, _mr); break;
  }
  if (is_nullable) {
    _null_mask = cudf::detail::create_null_mask(
      size, mask_state::ALL_NULL, rmm::cuda_stream_view(stream), _mr);
  }
}

template <class string_policy>
string_policy column_buffer_base<string_policy>::empty_like(string_policy const& input)
{
  auto new_buff = string_policy(input.type, input.is_nullable);
  copy_buffer_data(input, new_buff);
  return new_buff;
}

template <class string_policy>
std::unique_ptr<column> make_column(column_buffer_base<string_policy>& buffer,
                                    column_name_info* schema_info,
                                    std::optional<reader_column_schema> const& schema,
                                    rmm::cuda_stream_view stream)
{
  if (schema_info != nullptr) { schema_info->name = buffer.name; }

  switch (buffer.type.id()) {
    case type_id::STRING:
      if (schema.value_or(reader_column_schema{}).is_enabled_convert_binary_to_strings()) {
        if (schema_info != nullptr) {
          schema_info->children.push_back(column_name_info{"offsets"});
          schema_info->children.push_back(column_name_info{"chars"});
        }

        // make_strings_column allocates new memory, it does not simply move
        // from the inputs, so we need to pass it the memory resource given to
        // the buffer on construction so that the memory is allocated using the
        // resource that the calling code expected.
        return buffer.make_string_column(stream);
      } else {
        // convert to binary
        auto const string_col = buffer.make_string_column(stream);
        auto const num_rows   = string_col->size();
        auto const null_count = string_col->null_count();
        auto col_content      = string_col->release();

        // convert to uint8 column, strings are currently stored as int8
        auto contents =
          col_content.children[strings_column_view::chars_column_index].release()->release();
        auto data = contents.data.release();

        auto uint8_col = std::make_unique<column>(
          data_type{type_id::UINT8}, data->size(), std::move(*data), rmm::device_buffer{}, 0);

        if (schema_info != nullptr) {
          schema_info->children.push_back(column_name_info{"offsets"});
          schema_info->children.push_back(column_name_info{"binary"});
        }

        return make_lists_column(
          num_rows,
          std::move(col_content.children[strings_column_view::offsets_column_index]),
          std::move(uint8_col),
          null_count,
          std::move(*col_content.null_mask));
      }

    case type_id::LIST: {
      // make offsets column
      auto offsets = std::make_unique<column>(
        data_type{type_id::INT32}, buffer.size, std::move(buffer._data), rmm::device_buffer{}, 0);

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
      auto child = make_column<string_policy>(buffer.children[0], child_info, child_schema, stream);

      // make the final list column (note : size is the # of offsets, so our actual # of rows is 1
      // less)
      return make_lists_column(buffer.size - 1,
                               std::move(offsets),
                               std::move(child),
                               buffer._null_count,
                               std::move(buffer._null_mask),
                               stream,
                               buffer._mr);
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
          make_column<string_policy>(buffer.children[i], child_info, child_schema, stream));
      }

      return make_structs_column(buffer.size,
                                 std::move(output_children),
                                 buffer._null_count,
                                 std::move(buffer._null_mask),
                                 stream,
                                 buffer._mr);
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
template <class string_policy>
std::unique_ptr<column> empty_like(column_buffer_base<string_policy>& buffer,
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
      auto child =
        cudf::io::detail::empty_like<string_policy>(buffer.children[0], child_info, stream, mr);

      // make the final list column
      return make_lists_column(
        0, std::move(offsets), std::move(child), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
    } break;

    case type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> output_children;
      output_children.reserve(buffer.children.size());
      std::transform(buffer.children.begin(),
                     buffer.children.end(),
                     std::back_inserter(output_children),
                     [&](auto& col) {
                       column_name_info* child_info = nullptr;
                       if (schema_info != nullptr) {
                         schema_info->children.push_back(column_name_info{""});
                         child_info = &schema_info->children.back();
                       }
                       return cudf::io::detail::empty_like<string_policy>(
                         col, child_info, stream, mr);
                     });

      return make_structs_column(
        0, std::move(output_children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
    } break;

    default: return cudf::make_empty_column(buffer.type);
  }
}

using pointer_type = gather_column_buffer;
using string_type  = inline_column_buffer;

using pointer_column_buffer = column_buffer_base<pointer_type>;
using string_column_buffer  = column_buffer_base<string_type>;

template std::unique_ptr<column> make_column<string_type>(
  string_column_buffer& buffer,
  column_name_info* schema_info,
  std::optional<reader_column_schema> const& schema,
  rmm::cuda_stream_view stream);

template std::unique_ptr<column> make_column<pointer_type>(
  pointer_column_buffer& buffer,
  column_name_info* schema_info,
  std::optional<reader_column_schema> const& schema,
  rmm::cuda_stream_view stream);

template std::unique_ptr<column> empty_like<string_type>(string_column_buffer& buffer,
                                                         column_name_info* schema_info,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr);

template std::unique_ptr<column> empty_like<pointer_type>(pointer_column_buffer& buffer,
                                                          column_name_info* schema_info,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::mr::device_memory_resource* mr);

template class column_buffer_base<pointer_type>;
template class column_buffer_base<string_type>;
}  // namespace cudf::io::detail
