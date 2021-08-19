/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

namespace cudf {
namespace io {
namespace detail {
/**
 * @brief Creates a `device_buffer` for holding `column` data.
 *
 * @param type The intended data type to populate
 * @param size The number of elements to be represented by the mask
 * @param state The desired state of the mask
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 *
 * @return `rmm::device_buffer` Device buffer allocation
 */
inline rmm::device_buffer create_data(
  data_type type,
  size_type size,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  std::size_t data_size = size_of(type) * size;

  rmm::device_buffer data(data_size, stream, mr);
  CUDA_TRY(cudaMemsetAsync(data.data(), 0, data_size, stream.value()));

  return data;
}

using string_index_pair = thrust::pair<const char*, size_type>;

/**
 * @brief Class for holding device memory buffers to column data that eventually
 * will be used to create a column.
 */
struct column_buffer {
  column_buffer() = default;

  // construct without a known size. call create() later to actually
  // allocate memory
  column_buffer(data_type _type, bool _is_nullable) : type(_type), is_nullable(_is_nullable) {}

  // construct with a known size. allocates memory
  column_buffer(data_type _type,
                size_type _size,
                bool _is_nullable                   = true,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : type(_type), is_nullable(_is_nullable)
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
              rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
              rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  auto data() { return _strings ? _strings->data() : _data.data(); }
  auto data_size() const { return _strings ? _strings->size() : _data.size(); }

  template <typename T = uint32_t>
  auto null_mask()
  {
    return static_cast<T*>(_null_mask.data());
  }
  auto null_mask_size() { return _null_mask.size(); };

  auto& null_count() { return _null_count; }

  std::unique_ptr<rmm::device_uvector<string_index_pair>> _strings;
  rmm::device_buffer _data{};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};

  data_type type{type_id::EMPTY};
  bool is_nullable{false};
  size_type size{0};
  std::vector<column_buffer> children;
  uint32_t user_data{0};  // arbitrary user data
  std::string name;
};

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
  column_name_info* schema_info       = nullptr,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates an equivalent empty column from an existing set of device memory buffers.
 *
 * This function preserves nested column type information by producing complete/identical
 * column hierarchies.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
std::unique_ptr<column> empty_like(
  column_buffer& buffer,
  column_name_info* schema_info       = nullptr,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace io
}  // namespace cudf
