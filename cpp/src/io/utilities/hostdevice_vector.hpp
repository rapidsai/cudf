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

#pragma once

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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

  hostdevice_vector() {}

  hostdevice_vector(hostdevice_vector &&v) { move(std::move(v)); }
  hostdevice_vector &operator=(hostdevice_vector &&v)
  {
    move(std::move(v));
    return *this;
  }

  explicit hostdevice_vector(size_t max_size,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default)
    : hostdevice_vector(max_size, max_size, stream)
  {
  }

  explicit hostdevice_vector(size_t initial_size,
                             size_t max_size,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default)
    : num_elements(initial_size), max_elements(max_size)
  {
    if (max_elements != 0) {
      CUDA_TRY(cudaMallocHost(&h_data, sizeof(T) * max_elements));
      d_data.resize(sizeof(T) * max_elements, stream);
    }
  }

  ~hostdevice_vector()
  {
    if (max_elements != 0) {
      auto const free_result = cudaFreeHost(h_data);
      assert(free_result == cudaSuccess);
    }
  }

  bool insert(const T &data)
  {
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
  T *device_ptr(size_t offset = 0) { return reinterpret_cast<T *>(d_data.data()) + offset; }
  T const *device_ptr(size_t offset = 0) const
  {
    return reinterpret_cast<T const *>(d_data.data()) + offset;
  }

  void host_to_device(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    CUDA_TRY(cudaMemcpyAsync(
      d_data.data(), h_data, memory_size(), cudaMemcpyHostToDevice, stream.value()));
    if (synchronize) { stream.synchronize(); }
  }

  void device_to_host(rmm::cuda_stream_view stream, bool synchronize = false)
  {
    CUDA_TRY(cudaMemcpyAsync(
      h_data, d_data.data(), memory_size(), cudaMemcpyDeviceToHost, stream.value()));
    if (synchronize) { stream.synchronize(); }
  }

 private:
  void move(hostdevice_vector &&v)
  {
    stream       = v.stream;
    max_elements = v.max_elements;
    num_elements = v.num_elements;
    h_data       = v.h_data;
    d_data       = std::move(v.d_data);

    v.max_elements = 0;
    v.num_elements = 0;
    v.h_data       = nullptr;
  }

  rmm::cuda_stream_view stream{};
  size_t max_elements{};
  size_t num_elements{};
  T *h_data{};
  rmm::device_buffer d_data{};
};
