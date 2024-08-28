/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf_test/default_stream.hpp>

#include <cudf/detail/utilities/stacktrace.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <iostream>

namespace cudf::test {

/**
 * @brief Resource that verifies that the default stream is not used in any allocation.
 */
class stream_checking_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  /**
   * @brief Construct a new adaptor.
   *
   * @throws `cudf::logic_error` if `upstream == nullptr`
   *
   * @param upstream The resource used for allocating/deallocating device memory
   */
  stream_checking_resource_adaptor(rmm::device_async_resource_ref upstream,
                                   bool error_on_invalid_stream,
                                   bool check_default_stream)
    : upstream_{upstream},
      error_on_invalid_stream_{error_on_invalid_stream},
      check_default_stream_{check_default_stream}
  {
  }

  stream_checking_resource_adaptor()                                                   = delete;
  ~stream_checking_resource_adaptor() override                                         = default;
  stream_checking_resource_adaptor(stream_checking_resource_adaptor const&)            = delete;
  stream_checking_resource_adaptor& operator=(stream_checking_resource_adaptor const&) = delete;
  stream_checking_resource_adaptor(stream_checking_resource_adaptor&&) noexcept        = default;
  stream_checking_resource_adaptor& operator=(stream_checking_resource_adaptor&&) noexcept =
    default;

  /**
   * @brief Returns the wrapped upstream resource
   *
   * @return The wrapped upstream resource
   */
  [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept
  {
    return upstream_;
  }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource as long as it fits inside the allocation limit.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   * by the upstream resource.
   * @throws `cudf::logic_error` if attempted on a default stream
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    verify_stream(stream);
    return upstream_.allocate_async(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, stream);
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `ptr`
   *
   * @throws `cudf::logic_error` if attempted on a default stream
   *
   * @param ptr Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override
  {
    verify_stream(stream);
    upstream_.deallocate_async(ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @param other The other resource to compare to
   * @return Whether or not the two resources are equivalent
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    if (this == &other) { return true; }
    auto cast = dynamic_cast<stream_checking_resource_adaptor const*>(&other);
    if (cast == nullptr) { return false; }
    return get_upstream_resource() == cast->get_upstream_resource();
  }

  /**
   * @brief Throw an error if the provided stream is invalid.
   *
   * A stream is invalid if:
   * - check_default_stream_ is true and this function is passed one of CUDA's
   *   default stream specifiers, or
   * - check_default_stream_ is false and this function is passed any stream
   *   other than the result of cudf::test::get_default_stream().
   *
   * @throws `std::runtime_error` if provided an invalid stream
   */
  void verify_stream(rmm::cuda_stream_view const stream) const
  {
    auto cstream{stream.value()};
    auto const invalid_stream =
      check_default_stream_ ? ((cstream == cudaStreamDefault) || (cstream == cudaStreamLegacy) ||
                               (cstream == cudaStreamPerThread))
                            : (cstream != cudf::test::get_default_stream().value());

    if (invalid_stream) {
      // Exclude the current function from stacktrace.
      std::cout << cudf::detail::get_stacktrace(cudf::detail::capture_last_stackframe::NO)
                << std::endl;

      if (error_on_invalid_stream_) {
        throw std::runtime_error("Attempted to perform an operation on an unexpected stream!");
      } else {
        std::cout << "Attempted to perform an operation on an unexpected stream!" << std::endl;
      }
    }
  }

  rmm::device_async_resource_ref
    upstream_;                    // the upstream resource used for satisfying allocation requests
  bool error_on_invalid_stream_;  // If true, throw an exception when the wrong stream is detected.
                                  // If false, simply print to stdout.
  bool check_default_stream_;  // If true, throw an exception when the default stream is observed.
                               // If false, throw an exception when anything other than
                               // cudf::test::get_default_stream() is observed.
};

}  // namespace cudf::test
