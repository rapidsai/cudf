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
 * @file column_buffer.hpp
 * @brief cuDF-IO Column-backing buffer utilities
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_test/column_utilities.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {
namespace detail {
/**
 * @brief Creates a `device_buffer` for holding `column` data.
 *
 * @param type The intended data type to populate
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned device_buffer
 *
 * @return `rmm::device_buffer` Device buffer allocation
 */
inline rmm::device_buffer create_data(
  data_type type,
  size_type size,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  std::size_t data_size = size_of(type) * size;

  rmm::device_buffer data(data_size, stream, mr);
  CUDA_TRY(cudaMemsetAsync(data.data(), 0, data_size, stream));

  return data;
}

/**
 * @brief Class for holding device memory buffers to column data that eventually
 * will be used to create a column.
 */
struct column_buffer {
  // there is a potential bug here.  In the decoding step, the buffer of
  // data holding these pairs is cast to an nvstrdesc_s, which is a struct
  // containing <const char *, size_t>.   So there is a mismatch between the
  // size_type and the size_t.  I believe this works because the str_pair is
  // aligned out to 8 bytes anyway.
  using str_pair = thrust::pair<const char*, size_type>;

  column_buffer() = default;

  // construct without a known size. call create() later to actually
  // allocate memory
  column_buffer(data_type _type, bool _is_nullable) : type(_type), is_nullable(_is_nullable) {}

  // construct with a known size. allocates memory
  column_buffer(data_type _type,
                size_type _size,
                bool _is_nullable                   = true,
                cudaStream_t stream                 = 0,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : type(_type), is_nullable(_is_nullable), _null_count(0)
  {
    create(_size, stream, mr);
  }

  // move constructor
  column_buffer(column_buffer&& col) = default;
  column_buffer& operator=(column_buffer&& col) = default;

  // copy constructor
  column_buffer(column_buffer const& col) = delete;
  column_buffer& operator=(column_buffer const& col) = delete;

  // instantiate a column of known type with a specified size.  Allows deferred creation for
  // preprocessing steps such as in the Parquet reader
  void create(size_type _size,
              cudaStream_t stream                 = 0,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    size = _size;

    switch (type.id()) {
      case type_id::STRING: _strings.resize(size); break;

      // list columns store a buffer of int32's as offsets to represent
      // their individual rows
      case type_id::LIST: _data = create_data(data_type{type_id::INT32}, size, stream, mr); break;

      // struct columns store no data themselves.  just validity and children.
      case type_id::STRUCT: break;

      default: _data = create_data(type, size, stream, mr); break;
    }
    if (is_nullable) { _null_mask = create_null_mask(size, mask_state::ALL_NULL, stream, mr); }
  }

  auto data() { return _strings.size() ? _strings.data().get() : _data.data(); }
  auto data_size() { return std::max(_data.size(), _strings.size() * sizeof(str_pair)); }

  template <typename T = uint32_t>
  auto null_mask()
  {
    return static_cast<T*>(_null_mask.data());
  }
  auto null_mask_size() { return _null_mask.size(); };

  auto& null_count() { return _null_count; }

  rmm::device_vector<str_pair> _strings;
  rmm::device_buffer _data{};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};

  bool is_nullable{false};
  data_type type{type_id::EMPTY};
  size_type size{0};
  std::vector<column_buffer> children;
  uint32_t user_data{0};  // arbitrary user data
  std::string name;
};

namespace {
/**
 * @brief Creates a column from an existing set of device memory buffers.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
std::unique_ptr<column> make_column(
  column_buffer& buffer,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  column_name_info* schema_info       = nullptr)
{
  using str_pair = thrust::pair<const char*, size_type>;

  if (schema_info != nullptr) { schema_info->name = buffer.name; }

  switch (buffer.type.id()) {
    case type_id::STRING:
      if (schema_info != nullptr) {
        schema_info->children.push_back(column_name_info{"offsets"});
        schema_info->children.push_back(column_name_info{"chars"});
      }
      return make_strings_column(buffer._strings, stream, mr);

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

      // make child column
      CUDF_EXPECTS(buffer.children.size() > 0, "Encountered malformed column_buffer");
      auto child = make_column(buffer.children[0], stream, mr, child_info);

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
      std::transform(buffer.children.begin(),
                     buffer.children.end(),
                     std::back_inserter(output_children),
                     [&](column_buffer& col) {
                       column_name_info* child_info = nullptr;
                       if (schema_info != nullptr) {
                         schema_info->children.push_back(column_name_info{""});
                         child_info = &schema_info->children.back();
                       }
                       return make_column(col, stream, mr, child_info);
                     });

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

}  // namespace

}  // namespace detail
}  // namespace io
}  // namespace cudf
