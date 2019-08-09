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

#pragma once

#include <cudf/cudf.h>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>

#include <cuda_runtime.h>
#include <rmm/rmm.h>

#include <algorithm>
#include <cstring>

/**
 * @brief A helper class that wraps a gdf_column and any associated memory.
 *
 * This abstraction initializes and manages a gdf_column (fields and memory)
 * while still allowing direct access. Memory is automatically deallocated
 * unless ownership is transferred via releasing and assigning the raw pointer.
 **/
class gdf_column_wrapper {
 public:
  gdf_column_wrapper(gdf_size_type size, gdf_dtype dtype,
                     gdf_dtype_extra_info dtype_info, const std::string name) {
    col = (gdf_column *)malloc(gdf_column_sizeof());
    gdf_column_view_augmented(col, nullptr, nullptr, size, dtype, 0, dtype_info,
                              name.c_str());
  }

  ~gdf_column_wrapper() {
    if (col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      free(col->col_name);
    }
    free(col);
  };

  gdf_column_wrapper(const gdf_column_wrapper &other) = delete;
  gdf_column_wrapper(gdf_column_wrapper &&other) : col(other.col) {
    other.col = nullptr;
  }

  gdf_error allocate() {
    // For strings, just store the ptr + length. Eventually, column's data ptr
    // is replaced with an NvString instance created from these pairs.
    const auto num_rows = std::max(col->size, 1);
    const auto column_byte_width = (col->dtype == GDF_STRING)
                                       ? sizeof(std::pair<const char *, size_t>)
                                       : cudf::byte_width(*col);

    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
    RMM_TRY(RMM_ALLOC(&col->valid, gdf_valid_allocation_size(num_rows), 0));
    CUDA_TRY(cudaMemset(col->valid, 0, gdf_valid_allocation_size(num_rows)));

    return GDF_SUCCESS;
  }

  gdf_column *operator->() const noexcept { return col; }
  gdf_column *get() const noexcept { return col; }
  gdf_column *release() noexcept {
    auto temp = col;
    col = nullptr;
    return temp;
  }

 private:
  gdf_column *col = nullptr;
};

/**
 * @brief A helper class that wraps fixed-length device memory for the GPU, and
 * a mirror host pinned memory for the CPU.
 *
 * This abstraction allocates a specified fixed chunk of device memory that can
 * initialized upfront, or gradually initialized as required.
 * The host-side memory can be used to manipulate data on the CPU before and
 * after operating on the same data on the GPU.
 **/
template <typename T>
class hostdevice_vector {
 public:
  using value_type = T;

  explicit hostdevice_vector(size_t max_size)
      : hostdevice_vector(max_size, max_size) {}

  explicit hostdevice_vector(size_t initial_size, size_t max_size)
      : num_elements(initial_size), max_elements(max_size) {
    CUDA_TRY(cudaMallocHost(&h_data, sizeof(T) * max_elements));
    RMM_ALLOC(&d_data, sizeof(T) * max_elements, 0);
  }

  ~hostdevice_vector() {
    RMM_FREE(d_data, 0);
    cudaFreeHost(h_data);
  }

  bool insert(const T &data) {
    if (num_elements < max_elements) {
      h_data[num_elements] = data;
      num_elements++;
      return true;
    }
    return false;
  }

  size_t max_size() const noexcept { return max_elements; }
  size_t size() const noexcept { return num_elements; }
  size_t memory_size() const noexcept { return sizeof(T) * num_elements; }

  T &operator[](size_t i) const { return h_data[i]; }
  T *host_ptr(size_t offset = 0) const { return h_data + offset; }
  T *device_ptr(size_t offset = 0) const { return d_data + offset; }

 private:
  size_t max_elements = 0;
  size_t num_elements = 0;
  T *h_data = nullptr;
  T *d_data = nullptr;
};

/**
 * @brief A helper class that owns a resizable device memory buffer.
 *
 * Memory in the allocated buffer is not initialized in the constructors.
 * Copy construction and copy assignment are disabled to prevent
 * accidental copies.
 **/
template <typename T>
class device_buffer {
  T* d_data_ = nullptr;
  size_t count_ = 0;
  cudaStream_t stream_ = 0;

public:
  device_buffer() noexcept = default;
  device_buffer(size_t cnt, cudaStream_t stream = 0):
    stream_(stream){
    resize(cnt);
  }

  T* data() const noexcept {return d_data_;}
  size_t size() const noexcept {return count_;}
  bool empty() const noexcept {return count_ == 0;}

  void resize(size_t cnt) {
    if (cnt == count_) {
      return;
    }
    // new size is zero, free the buffer if not null
    if(cnt == 0 && d_data_ != nullptr) {
      RMM_FREE(d_data_, stream_);
      d_data_ = nullptr;
      count_ = cnt;
      return;
    }

    T* new_ptr = nullptr;
    const auto error = RMM_ALLOC(&new_ptr, cnt*sizeof(T), stream_);
    if(error != RMM_SUCCESS) {
      cudf::detail::throw_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
    }
    // Copy to the new buffer, if some memory was already allocated
    if (count_ != 0) {
      const size_t copy_bytes = std::min(cnt, count_)*sizeof(T);
      CUDA_TRY(cudaMemcpyAsync(new_ptr, d_data_, copy_bytes, cudaMemcpyDefault, stream_));
      RMM_FREE(d_data_, stream_);
    }

    d_data_ = new_ptr;
    count_ = cnt;
  }

  device_buffer(device_buffer& ) = delete;
  device_buffer(device_buffer&& rh) noexcept {
    d_data_ = rh.d_data_; 
    count_ = rh.count_;
    stream_ = rh.stream_;

    rh.d_data_ = nullptr;
    rh.count_ = 0;
  }

  device_buffer& operator=(device_buffer& ) = delete;
  device_buffer& operator=(device_buffer&& rh) noexcept {
    RMM_FREE(d_data_, stream_);

    d_data_ = rh.d_data_; 
    count_ = rh.count_;
    stream_ = rh.stream_;

    rh.d_data_ = nullptr;
    rh.count_ = 0;
   
    return *this;
  }

  ~device_buffer() {
    RMM_FREE(d_data_, stream_);
  }
};
