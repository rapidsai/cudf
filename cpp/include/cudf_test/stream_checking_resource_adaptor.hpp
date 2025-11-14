/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device_memory_resource.hpp>

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
    return upstream_.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
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
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
  {
    verify_stream(stream);
    upstream_.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
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
