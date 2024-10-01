/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <fmt/core.h>
#include <nvtx3/nvtx3.hpp>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string_view>

namespace RMM_NAMESPACE {
namespace mr {
/**
 * @addtogroup device_resource_adaptors
 * @{
 * @file
 */
/**
 * @brief Resource that uses `Upstream` to allocate memory and logs information
 * about the requested allocation/deallocations.
 *
 * An instance of this resource can be constructed with an existing, upstream
 * resource in order to satisfy allocation requests and log
 * allocation/deallocation activity.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class nvtx_resource_adaptor final : public device_memory_resource {
 public:
 Upstream* upstream_backup{nullptr};
  /**
   * @brief Construct a new nvtx resource adaptor using `upstream` to satisfy
   * allocation requests and nvtx information about each allocation/free to
   * the file specified by `filename`.
   *
   * The logfile will be written using CSV formatting.
   *
   * Clears the contents of `filename` if it already exists.
   *
   * Creating multiple `nvtx_resource_adaptor`s with the same `filename` will
   * result in undefined behavior.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  nvtx_resource_adaptor(Upstream* upstream)
    : upstream_{to_device_async_resource_ref_checked(upstream)}
  {
    upstream_backup = upstream;
    nvtx3::mark("nvtx mr creation");
  }

  /**
   * @brief Construct a new nvtx resource adaptor using `upstream` to satisfy
   * allocation requests and nvtx information about each allocation/free to
   * the ostream specified by `stream`.
   *
   * The logfile will be written using CSV formatting.
   *
   * @throws rmm::logic_error if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param stream The ostream to write log info.
   */
  nvtx_resource_adaptor(Upstream* upstream, std::ostream& stream)
    : upstream_{to_device_async_resource_ref_checked(upstream)}
  {
    nvtx3::mark("nvtx mr creation with stream");
  }

  /**
   * @brief Construct a new nvtx resource adaptor using `upstream` to satisfy
   * allocation requests and nvtx information about each allocation/free to
   * the ostream specified by `stream`.
   *
   * The logfile will be written using CSV formatting.
   *
   * @param upstream The resource_ref used for allocating/deallocating device memory.
   */
  nvtx_resource_adaptor(device_async_resource_ref upstream)
    : upstream_{upstream}
  {
    nvtx3::mark("nvtx mr creation with stream");
  }

  nvtx_resource_adaptor()                                           = delete;
  ~nvtx_resource_adaptor() override                                 = default;
  nvtx_resource_adaptor(nvtx_resource_adaptor const&)            = delete;
  nvtx_resource_adaptor& operator=(nvtx_resource_adaptor const&) = delete;
  nvtx_resource_adaptor(nvtx_resource_adaptor&&) noexcept =
    default;  ///< @default_move_constructor
  nvtx_resource_adaptor& operator=(nvtx_resource_adaptor&&) noexcept =
    default;  ///< @default_move_assignment{nvtx_resource_adaptor}

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }
  /**
   * @brief Return the CSV header string
   *
   * @return CSV formatted header string of column names
   */
  [[nodiscard]] std::string header() const
  {
    return std::string{"Thread,Time,Action,Pointer,Size,Stream"};
  }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource and logs the allocation.
   *
   * If the upstream allocation is successful, logs the following CSV formatted
   * line to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"allocate",*pointer*,*bytes*,*stream*
   * ```
   *
   * If the upstream allocation failed, logs the following CSV formatted line
   * to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"allocate failure",0x0,*bytes*,*stream*
   * ```
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws rmm::bad_alloc if the requested allocation could not be fulfilled
   * by the upstream resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    try {
      auto const ptr = get_upstream_resource().allocate_async(bytes, stream);
    //   nvtx3::mark(fmt::format("allocate,{},{},{}", ptr, bytes, fmt::ptr(stream.value())));
      using stat_mr_t = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
      if(auto smr = dynamic_cast<stat_mr_t*>(upstream_backup); smr != nullptr) {
        auto counts = smr->get_bytes_counter();
        nvtx3::mark(nvtx3::rgb{0, 200, 0},fmt::format("value: {:_>6} MB, peak: {:_>6} MB, allo {:_>12} {:>4.1f}GB", (counts.value)>>20, (counts.peak)>>20, bytes, bytes/(1024*1024*1024.0) ));
        std::cout<<"+"<<bytes<<std::endl;
      }
      return ptr;
    } catch (...) {
      nvtx3::mark(fmt::format("allocate failure,{},{},{}", nullptr, bytes, fmt::ptr(stream.value())));
      throw;
    }
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `ptr` and log the
   * deallocation.
   *
   * Every invocation of `nvtx_resource_adaptor::do_deallocate` will write
   * the following CSV formatted line to the file specified at construction:
   * ```
   * thread_id,*TIMESTAMP*,"free",*bytes*,*stream*
   * ```
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    // nvtx3::mark(fmt::format("free,{},{},{}", ptr, bytes, fmt::ptr(stream.value())));
    using stat_mr_t = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
    if(auto smr = dynamic_cast<stat_mr_t*>(upstream_backup); smr != nullptr) {
      auto counts = smr->get_bytes_counter();
      nvtx3::mark(nvtx3::rgb{200, 0, 0}, fmt::format("value: {:_>6} MB, peak: {:_>6} MB, free {:_>12} {:>4.1f}GB", (counts.value)>>20, (counts.peak)>>20, bytes, bytes/(1024*1024*1024.0)));
      std::cout<<"-"<<bytes<<std::endl;
    }
    get_upstream_resource().deallocate_async(ptr, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto const* cast = dynamic_cast<nvtx_resource_adaptor<Upstream> const*>(&other);
    if (cast == nullptr) { return false; }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  device_async_resource_ref upstream_;  ///< The upstream resource used for satisfying
                                        ///< allocation requests
};

/**
 * @brief Convenience factory to return a `nvtx_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @param filename Name of the file to write log info. If not specified,
 * retrieves the log file name from the environment variable "RMM_LOG_FILE".
 * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
 * performance.
 * @return The new nvtx resource adaptor
 */
template <typename Upstream>
[[deprecated(
  "make_nvtx_adaptor is deprecated in RMM 24.10. Use the nvtx_resource_adaptor constructor "
  "instead.")]]
nvtx_resource_adaptor<Upstream> make_nvtx_adaptor(
  Upstream* upstream)
{
  return nvtx_resource_adaptor<Upstream>{upstream};
}

/**
 * @brief Convenience factory to return a `nvtx_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @param stream The ostream to write log info.
 * @param auto_flush If true, flushes the log for every (de)allocation. Warning, this will degrade
 * performance.
 * @return The new nvtx resource adaptor
 */
template <typename Upstream>
[[deprecated(
  "make_nvtx_adaptor is deprecated in RMM 24.10. Use the nvtx_resource_adaptor constructor "
  "instead.")]]
nvtx_resource_adaptor<Upstream> make_nvtx_adaptor(Upstream* upstream,
                                                        std::ostream& stream)
{
  return nvtx_resource_adaptor<Upstream>{upstream, stream};
}

/** @} */  // end of group
}  // namespace mr
}  // namespace RMM_NAMESPACE
