/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/**
 * @brief Convenience factories for creating device vectors from host spans
 * @file vector_factories.hpp
 */

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <vector>

namespace cudf {
namespace detail {

/**
 * @brief Asynchronously construct a `device_uvector` containing a deep copy of data from a
 * `host_span`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The host_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename T>
rmm::device_uvector<T> make_device_uvector_async(
  host_span<T const> source_data,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  rmm::device_uvector<T> ret(source_data.size(), stream, mr);
  CUDA_TRY(cudaMemcpyAsync(ret.data(),
                           source_data.data(),
                           source_data.size() * sizeof(T),
                           cudaMemcpyDefault,
                           stream.value()));
  return ret;
}

/**
 * @brief Asynchronously construct a `device_uvector` containing a deep copy of data from a host
 * container
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input host container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename Container,
          std::enable_if_t<
            std::is_convertible<Container,
                                host_span<typename Container::value_type const>>::value>* = nullptr>
rmm::device_uvector<typename Container::value_type> make_device_uvector_async(
  Container const& c,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return make_device_uvector_async(host_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Asynchronously construct a `device_uvector` containing a deep copy of data from a
 * `device_span`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The device_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename T>
rmm::device_uvector<T> make_device_uvector_async(
  device_span<T const> source_data,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  rmm::device_uvector<T> ret(source_data.size(), stream, mr);
  CUDA_TRY(cudaMemcpyAsync(ret.data(),
                           source_data.data(),
                           source_data.size() * sizeof(T),
                           cudaMemcpyDefault,
                           stream.value()));
  return ret;
}

/**
 * @brief Asynchronously construct a `device_uvector` containing a deep copy of data from a device
 * container
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input device container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <
  typename Container,
  std::enable_if_t<
    std::is_convertible<Container, device_span<typename Container::value_type const>>::value>* =
    nullptr>
rmm::device_uvector<typename Container::value_type> make_device_uvector_async(
  Container const& c,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return make_device_uvector_async(
    device_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Synchronously construct a `device_uvector` containing a deep copy of data from a
 * `host_span`
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The host_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename T>
rmm::device_uvector<T> make_device_uvector_sync(
  host_span<T const> source_data,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto ret = make_device_uvector_async(source_data, stream, mr);
  stream.synchronize();
  return ret;
}

/**
 * @brief Synchronously construct a `device_uvector` containing a deep copy of data from a host
 * container
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input host container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename Container,
          std::enable_if_t<
            std::is_convertible<Container,
                                host_span<typename Container::value_type const>>::value>* = nullptr>
rmm::device_uvector<typename Container::value_type> make_device_uvector_sync(
  Container const& c,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return make_device_uvector_sync(host_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Synchronously construct a `device_uvector` containing a deep copy of data from a
 * `device_span`
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The device_span of data to deep copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <typename T>
rmm::device_uvector<T> make_device_uvector_sync(
  device_span<T const> source_data,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto ret = make_device_uvector_async(source_data, stream, mr);
  stream.synchronize();
  return ret;
}

/**
 * @brief Synchronously construct a `device_uvector` containing a deep copy of data from a device
 * container
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input device container from which to copy
 * @param stream The stream on which to allocate memory and perform the copy
 * @param mr The memory resource to use for allocating the returned device_uvector
 * @return A device_uvector containing the copied data
 */
template <
  typename Container,
  std::enable_if_t<
    std::is_convertible<Container, device_span<typename Container::value_type const>>::value>* =
    nullptr>
rmm::device_uvector<typename Container::value_type> make_device_uvector_sync(
  Container const& c,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return make_device_uvector_sync(device_span<typename Container::value_type const>{c}, stream, mr);
}

/**
 * @brief Asynchronously construct a `std::vector` containing a copy of data from a
 * `device_span`
 *
 * @note This function does not synchronize `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The device data to copy
 * @param stream The stream on which to perform the copy
 * @return The data copied to the host
 */
template <typename T>
std::vector<T> make_std_vector_async(device_span<T const> v,
                                     rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  std::vector<T> result(v.size());
  CUDA_TRY(cudaMemcpyAsync(
    result.data(), v.data(), v.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.value()));
  return result;
}

/**
 * @brief Asynchronously construct a `std::vector` containing a copy of data from a device
 * container
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input device container from which to copy
 * @param stream The stream on which to perform the copy
 * @return The data copied to the host
 */
template <
  typename Container,
  std::enable_if_t<
    std::is_convertible<Container, device_span<typename Container::value_type const>>::value>* =
    nullptr>
std::vector<typename Container::value_type> make_std_vector_async(
  Container const& c, rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  return make_std_vector_async(device_span<typename Container::value_type const>{c}, stream);
}

/**
 * @brief Synchronously construct a `std::vector` containing a copy of data from a
 * `device_span`
 *
 * @note This function does a synchronize on `stream`.
 *
 * @tparam T The type of the data to copy
 * @param source_data The device data to copy
 * @param stream The stream on which to perform the copy
 * @return The data copied to the host
 */
template <typename T>
std::vector<T> make_std_vector_sync(device_span<T const> v,
                                    rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  auto result = make_std_vector_async(v, stream);
  stream.synchronize();
  return result;
}

/**
 * @brief Synchronously construct a `std::vector` containing a copy of data from a device
 * container
 *
 * @note This function synchronizes `stream`.
 *
 * @tparam Container The type of the container to copy from
 * @tparam T The type of the data to copy
 * @param c The input device container from which to copy
 * @param stream The stream on which to perform the copy
 * @return The data copied to the host
 */
template <
  typename Container,
  std::enable_if_t<
    std::is_convertible<Container, device_span<typename Container::value_type const>>::value>* =
    nullptr>
std::vector<typename Container::value_type> make_std_vector_sync(
  Container const& c, rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  return make_std_vector_sync(device_span<typename Container::value_type const>{c}, stream);
}

}  // namespace detail

}  // namespace cudf
