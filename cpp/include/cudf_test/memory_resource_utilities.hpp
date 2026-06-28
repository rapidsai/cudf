/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Exception-safe owner for a temporary current device resource.
 *
 * The installed resource and the previous resource are held by owning type-erased resource values.
 * Destruction restores the previous resource, including during stack unwinding. Because the current
 * resource is device-global state, scopes must not overlap concurrent work that changes or uses the
 * current resource.
 */
class scoped_current_device_resource {
 public:
  /**
   * @brief Install `resource` as the current device resource for this object's lifetime.
   *
   * @param resource Owning type-erased resource value to install
   */
  explicit scoped_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible> resource)
    : _previous{cudf::set_current_device_resource(std::move(resource))}
  {
  }

  ~scoped_current_device_resource()
  {
    auto replaced = cudf::set_current_device_resource(std::move(_previous));
    static_cast<void>(replaced);
  }

  scoped_current_device_resource(scoped_current_device_resource const&)            = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource const&) = delete;
  scoped_current_device_resource(scoped_current_device_resource&&)                 = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource&&)      = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> _previous;
};

/**
 * @brief Reusable memory-resource instrumentation for API routing tests.
 *
 * This harness owns independent statistics adaptors for setup, output, and temporary allocations.
 * It also owns an allocation-failing resource that can be installed as the current resource around
 * only the API call under test. Counter assertion helpers synchronize the supplied stream before
 * inspecting allocation state.
 */
class memory_resource_test_harness {
 public:
  /**
   * @brief Construct instrumentation over `upstream`.
   *
   * @param upstream Resource used by each independent statistics adaptor
   */
  explicit memory_resource_test_harness(
    rmm::device_async_resource_ref upstream = cudf::get_current_device_resource_ref())
    : _setup_mr{upstream},
      _output_mr{upstream},
      _temporary_mr{upstream},
      _failing_mr{[](std::size_t, rmm::cuda_stream_view, void*) -> void* {
                    throw rmm::bad_alloc{"Unexpected allocation from the current device resource"};
                  },
                  [](void*, std::size_t, rmm::cuda_stream_view, void*) {}}
  {
  }

  /** @brief Return the statistics resource used to construct test inputs and expected results. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& setup_mr() noexcept { return _setup_mr; }

  /** @brief Return the statistics resource used for API output allocations. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& output_mr() noexcept { return _output_mr; }

  /** @brief Return the statistics resource used for API temporary allocations. */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor& temporary_mr() noexcept
  {
    return _temporary_mr;
  }

  /**
   * @brief Return explicit output and temporary resources without consulting the current resource.
   */
  [[nodiscard]] cudf::memory_resources resources() noexcept
  {
    return cudf::memory_resources{_output_mr, _temporary_mr};
  }

  /**
   * @brief Install an allocation-failing current resource for the returned scope's lifetime.
   *
   * Keep the returned scope limited to the API invocation and stream synchronization. Construct
   * inputs and validate results outside this scope.
   *
   * @return Scope that restores the prior current resource on destruction
   */
  [[nodiscard]] scoped_current_device_resource fail_on_current_device_resource_use()
  {
    return scoped_current_device_resource{_failing_mr};
  }

  /** @brief Synchronize `stream` before leaving a failing-resource API scope. */
  void synchronize(rmm::cuda_stream_view stream = cudf::test::get_default_stream()) const
  {
    stream.synchronize();
  }

  /** @brief Assert that output allocations are live after synchronizing `stream`. */
  void expect_output_allocations_live(
    rmm::cuda_stream_view stream = cudf::test::get_default_stream()) const
  {
    synchronize(stream);
    EXPECT_GT(_output_mr.get_bytes_counter().value, 0);
  }

  /** @brief Assert that temporary allocations were made after synchronizing `stream`. */
  void expect_temporary_allocation_activity(
    rmm::cuda_stream_view stream = cudf::test::get_default_stream()) const
  {
    synchronize(stream);
    EXPECT_GT(_temporary_mr.get_bytes_counter().total, 0);
  }

  /** @brief Assert that no temporary allocations remain live after synchronizing `stream`. */
  void expect_temporary_allocations_released(
    rmm::cuda_stream_view stream = cudf::test::get_default_stream()) const
  {
    synchronize(stream);
    EXPECT_EQ(_temporary_mr.get_bytes_counter().value, 0);
  }

  /**
   * @brief Assert that no output or temporary allocations remain after synchronizing `stream`.
   *
   * Setup allocations are intentionally excluded because test inputs may remain alive while result
   * allocation lifetimes are checked.
   */
  void expect_no_live_allocations(
    rmm::cuda_stream_view stream = cudf::test::get_default_stream()) const
  {
    synchronize(stream);
    EXPECT_EQ(_output_mr.get_bytes_counter().value, 0);
    EXPECT_EQ(_temporary_mr.get_bytes_counter().value, 0);
  }

 private:
  rmm::mr::statistics_resource_adaptor _setup_mr;
  rmm::mr::statistics_resource_adaptor _output_mr;
  rmm::mr::statistics_resource_adaptor _temporary_mr;
  rmm::mr::callback_memory_resource _failing_mr;
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
