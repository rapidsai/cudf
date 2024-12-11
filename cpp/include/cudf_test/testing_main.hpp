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

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/cxxopts.hpp>
#include <cudf_test/stream_checking_resource_adaptor.hpp>

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

struct config {
  std::string rmm_mode;
  std::string stream_mode;
  std::string stream_error_mode;
};

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  auto const [free, total] = rmm::available_device_memory();
  auto const min_alloc =
    rmm::align_down(std::min(free, total / 10), rmm::CUDA_ALLOCATION_ALIGNMENT);
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda(), min_alloc);
}

inline auto make_arena()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(make_cuda());
}

inline auto make_binning()
{
  auto pool = make_pool();
  // Add a binning_memory_resource with fixed-size bins of sizes 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool, 18, 22);
  return mr;
}

/**
 * @brief Creates a memory resource for the unit test environment
 * given the name of the allocation mode.
 *
 * The returned resource instance must be kept alive for the duration of
 * the tests. Attaching the resource to a TestEnvironment causes
 * issues since the environment objects are not destroyed until
 * after the runtime is shutdown.
 *
 * @throw cudf::logic_error if the `allocation_mode` is unsupported.
 *
 * @param allocation_mode String identifies which resource type.
 *        Accepted types are "pool", "cuda", and "managed" only.
 * @return Memory resource instance
 */
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& allocation_mode)
{
  if (allocation_mode == "binning") return make_binning();
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "async") return make_async();
  if (allocation_mode == "pool") return make_pool();
  if (allocation_mode == "arena") return make_arena();
  if (allocation_mode == "managed") return make_managed();
  CUDF_FAIL("Invalid RMM allocation mode: " + allocation_mode);
}

}  // namespace test
}  // namespace CUDF_EXPORT cudf

/**
 * @brief Parses the cuDF test command line options.
 *
 * Currently only supports 'rmm_mode' string parameter, which set the rmm
 * allocation mode. The default value of the parameter is 'pool'.
 * Environment variable 'CUDF_TEST_RMM_MODE' can also be used to set the rmm
 * allocation mode. If both are set, the value of 'rmm_mode' string parameter
 * takes precedence.
 *
 * @return Parsing results in the form of unordered map
 */
inline auto parse_cudf_test_opts(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuDF tests command line options");
    char const* env_rmm_mode = std::getenv("GTEST_CUDF_RMM_MODE");  // Overridden by CLI options
    char const* env_stream_mode =
      std::getenv("GTEST_CUDF_STREAM_MODE");  // Overridden by CLI options
    char const* env_stream_error_mode =
      std::getenv("GTEST_CUDF_STREAM_ERROR_MODE");  // Overridden by CLI options
    auto default_rmm_mode          = env_rmm_mode ? env_rmm_mode : "pool";
    auto default_stream_mode       = env_stream_mode ? env_stream_mode : "default";
    auto default_stream_error_mode = env_stream_error_mode ? env_stream_error_mode : "error";
    options.allow_unrecognised_options().add_options()(
      "rmm_mode",
      "RMM allocation mode",
      cxxopts::value<std::string>()->default_value(default_rmm_mode));
    // `new_cudf_default` means that cudf::get_default_stream has been patched,
    // so we raise errors anywhere that a CUDA default stream is observed
    // instead of cudf::get_default_stream(). This corresponds to compiling
    // identify_stream_usage with STREAM_MODE_TESTING=OFF (must do both at the
    // same time).
    // `new_testing_default` means that cudf::test::get_default_stream has been
    // patched, so we raise errors anywhere that _any_ other stream is
    // observed. This corresponds to compiling identify_stream_usage with
    // STREAM_MODE_TESTING=ON (must do both at the same time).
    options.allow_unrecognised_options().add_options()(
      "stream_mode",
      "Whether to use a non-default stream",
      cxxopts::value<std::string>()->default_value(default_stream_mode));
    options.allow_unrecognised_options().add_options()(
      "stream_error_mode",
      "Whether to error or print to stdout when a non-default stream is observed and stream_mode "
      "is not \"default\"",
      cxxopts::value<std::string>()->default_value(default_stream_error_mode));
    return options.parse(argc, argv);
  } catch (cxxopts::OptionException const& e) {
    CUDF_FAIL("Error parsing command line options");
  }
}

/**
 * @brief Sets up stream mode memory resource adaptor
 *
 * The resource adaptor is only set as the current device resource if the
 * stream mode is enabled.
 *
 * The caller must keep the return object alive for the life of the test runs.
 *
 * @param cmd_opts Command line options returned by parse_cudf_test_opts
 * @return Memory resource adaptor
 */
inline auto make_memory_resource_adaptor(cudf::test::config const& config)
{
  auto resource = cudf::test::create_memory_resource(config.rmm_mode);
  cudf::set_current_device_resource(resource.get());
  return resource;
}

/**
 * @brief Sets up stream mode memory resource adaptor
 *
 * The resource adaptor is only set as the current device resource if the
 * stream mode is enabled.
 *
 * The caller must keep the return object alive for the life of the test runs.
 *
 * @param cmd_opts Command line options returned by parse_cudf_test_opts
 * @return Memory resource adaptor
 */
inline auto make_stream_mode_adaptor(cudf::test::config const& config)
{
  auto resource                      = cudf::get_current_device_resource_ref();
  auto const error_on_invalid_stream = (config.stream_error_mode == "error");
  auto const check_default_stream    = (config.stream_mode == "new_cudf_default");
  auto adaptor                       = cudf::test::stream_checking_resource_adaptor(
    resource, error_on_invalid_stream, check_default_stream);
  if ((config.stream_mode == "new_cudf_default") || (config.stream_mode == "new_testing_default")) {
    cudf::set_current_device_resource(&adaptor);
  }
  return adaptor;
}

/**
 * @brief Should be called in every test program that uses rmm allocators since it maintains the
 * lifespan of the rmm default memory resource. this function parses the command line to customize
 * test behavior, like the allocation mode used for creating the default memory resource.
 *
 */
inline void init_cudf_test(int argc, char** argv, cudf::test::config const& config_override = {})
{
  // static lifetime to keep rmm resource alive till tests end
  auto const cmd_opts       = parse_cudf_test_opts(argc, argv);
  cudf::test::config config = config_override;
  if (config.rmm_mode.empty()) { config.rmm_mode = cmd_opts["rmm_mode"].as<std::string>(); }

  if (config.stream_mode.empty()) {
    config.stream_mode = cmd_opts["stream_mode"].as<std::string>();
  }

  if (config.stream_error_mode.empty()) {
    config.stream_error_mode = cmd_opts["stream_error_mode"].as<std::string>();
  }

  [[maybe_unused]] static auto mr      = make_memory_resource_adaptor(config);
  [[maybe_unused]] static auto adaptor = make_stream_mode_adaptor(config);
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * This `main` function is a wrapper around the google test generated `main`,
 * maintaining the original functionality.
 */
#define CUDF_TEST_PROGRAM_MAIN()            \
  int main(int argc, char** argv)           \
  {                                         \
    ::testing::InitGoogleTest(&argc, argv); \
    init_cudf_test(argc, argv);             \
    return RUN_ALL_TESTS();                 \
  }
