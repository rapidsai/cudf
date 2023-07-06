/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/io/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/pair.h>

namespace cudf {
namespace io {
namespace detail {
/**
 * @brief Creates a `device_buffer` for holding `column` data.
 *
 * @param type The intended data type to populate
 * @param size The number of elements to be represented by the mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 *
 * @return `rmm::device_buffer` Device buffer allocation
 */
inline rmm::device_buffer create_data(data_type type,
                                      size_type size,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  std::size_t data_size = size_of(type) * size;

  rmm::device_buffer data(data_size, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(data.data(), 0, data_size, stream.value()));

  return data;
}

using string_index_pair = thrust::pair<char const*, size_type>;

// forward declare friend functions
template <typename string_policy>
class column_buffer_base;

/**
 * @brief Creates a column from an existing set of device memory buffers.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param schema_info Schema information for the column to write optionally.
 * @param schema Optional schema used to control string to binary conversions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
template <class string_policy>
std::unique_ptr<column> make_column(column_buffer_base<string_policy>& buffer,
                                    column_name_info* schema_info,
                                    std::optional<reader_column_schema> const& schema,
                                    rmm::cuda_stream_view stream);

template <typename string_policy>
class column_buffer_base {
 public:
  column_buffer_base() = default;

  // construct without a known size. call create() later to actually allocate memory
  column_buffer_base(data_type _type, bool _is_nullable) : type(_type), is_nullable(_is_nullable) {}

  column_buffer_base(data_type _type,
                     size_type _size,
                     bool _is_nullable,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr)
    : column_buffer_base(_type, _is_nullable)
  {
  }

  // move constructor
  column_buffer_base(column_buffer_base&& col)            = default;
  column_buffer_base& operator=(column_buffer_base&& col) = default;

  // copy constructor
  column_buffer_base(column_buffer_base const& col)            = delete;
  column_buffer_base& operator=(column_buffer_base const& col) = delete;

  // instantiate a column of known type with a specified size.  Allows deferred creation for
  // preprocessing steps such as in the Parquet reader
  void create(size_type _size, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr);

  // Create a new column_buffer that has empty data but with the same basic information as the
  // input column, including same type, nullability, name, and user_data.
  static string_policy empty_like(string_policy const& input);

  void set_null_mask(rmm::device_buffer&& mask) { _null_mask = std::move(mask); }

  template <typename T = uint32_t>
  auto null_mask()
  {
    return static_cast<T*>(_null_mask.data());
  }
  auto null_mask_size() { return _null_mask.size(); }
  auto& null_count() { return _null_count; }

  auto data() { return static_cast<string_policy*>(this)->data_impl(); }
  auto data() const { return static_cast<string_policy const*>(this)->data_impl(); }
  auto data_size() const { return static_cast<string_policy const*>(this)->data_size_impl(); }

  std::unique_ptr<column> make_string_column(rmm::cuda_stream_view stream)
  {
    return static_cast<string_policy*>(this)->make_string_column_impl(stream);
  }

 protected:
  rmm::device_buffer _data{};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};
  rmm::mr::device_memory_resource* _mr;

 public:
  data_type type{type_id::EMPTY};
  bool is_nullable{false};
  size_type size{0};
  uint32_t user_data{0};  // arbitrary user data
  std::string name;

  std::vector<string_policy> children;

  friend std::unique_ptr<column> make_column<string_policy>(
    column_buffer_base& buffer,
    column_name_info* schema_info,
    std::optional<reader_column_schema> const& schema,
    rmm::cuda_stream_view stream);
};

// column buffer that uses a string_index_pair for strings data, requiring a gather step when
// creating a string column
class gather_column_buffer : public column_buffer_base<gather_column_buffer> {
 public:
  gather_column_buffer() = default;

  // construct without a known size. call create() later to actually allocate memory
  gather_column_buffer(data_type _type, bool _is_nullable)
    : column_buffer_base<gather_column_buffer>(_type, _is_nullable)
  {
  }

  gather_column_buffer(data_type _type,
                       size_type _size,
                       bool _is_nullable,
                       rmm::cuda_stream_view stream,
                       rmm::mr::device_memory_resource* mr)
    : column_buffer_base<gather_column_buffer>(_type, _size, _is_nullable, stream, mr)
  {
    create(_size, stream, mr);
  }

  void allocate_strings_data(rmm::cuda_stream_view stream);

  void* data_impl() { return _strings ? _strings->data() : _data.data(); }
  void const* data_impl() const { return _strings ? _strings->data() : _data.data(); }
  size_t data_size_impl() const { return _strings ? _strings->size() : _data.size(); }

  std::unique_ptr<column> make_string_column_impl(rmm::cuda_stream_view stream);

 public:
  std::unique_ptr<rmm::device_uvector<string_index_pair>> _strings;
};

// column buffer that stores string data internally which can be passed directly when
// creating a string column
class inline_column_buffer : public column_buffer_base<inline_column_buffer> {
 public:
  inline_column_buffer() = default;

  // construct without a known size. call create() later to actually allocate memory
  inline_column_buffer(data_type _type, bool _is_nullable)
    : column_buffer_base<inline_column_buffer>(_type, _is_nullable)
  {
  }

  inline_column_buffer(data_type _type,
                       size_type _size,
                       bool _is_nullable,
                       rmm::cuda_stream_view stream,
                       rmm::mr::device_memory_resource* mr)
    : column_buffer_base<inline_column_buffer>(_type, _size, _is_nullable, stream, mr)
  {
    create(_size, stream, mr);
  }

  void allocate_strings_data(rmm::cuda_stream_view stream);

  void* data_impl() { return _data.data(); }
  void const* data_impl() const { return _data.data(); }
  size_t data_size_impl() const { return _data.size(); }
  std::unique_ptr<column> make_string_column_impl(rmm::cuda_stream_view stream);

  void create_string_data(size_t num_bytes, rmm::cuda_stream_view stream);
  void* string_data() { return _string_data.data(); }
  void const* string_data() const { return _string_data.data(); }
  size_t string_size() const { return _string_data.size(); }

 private:
  rmm::device_buffer _string_data{};
};

using column_buffer = gather_column_buffer;

/**
 * @brief Creates an equivalent empty column from an existing set of device memory buffers.
 *
 * This function preserves nested column type information by producing complete/identical
 * column hierarchies.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param schema_info Schema information for the column to write optionally.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
template <class string_policy>
std::unique_ptr<column> empty_like(column_buffer_base<string_policy>& buffer,
                                   column_name_info* schema_info,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace io
}  // namespace cudf
