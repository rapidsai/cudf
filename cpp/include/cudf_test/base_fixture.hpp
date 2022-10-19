/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <random>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/cxxopts.hpp>
#include <cudf_test/file_utilities.hpp>
#include <cudf_test/stream_checking_resource_adapter.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace cudf {
namespace test {
/**
 * @brief Base test fixture class from which all libcudf tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cudf::test::BaseFixture {};
 * ```
 */
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource* _mr{rmm::mr::get_current_device_resource()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   * @return pointer to memory resource
   */
  rmm::mr::device_memory_resource* mr() { return _mr; }
};

template <typename T, typename Enable = void>
struct uniform_distribution_impl {
};
template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = std::uniform_int_distribution<T>;
};

template <>
struct uniform_distribution_impl<bool> {
  using type = std::bernoulli_distribution;
};

template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
struct uniform_distribution_impl<
  T,
  std::enable_if_t<cudf::is_chrono<T>() or cudf::is_fixed_point<T>()>> {
  using type = std::uniform_int_distribution<typename T::rep>;
};

template <typename T>
using uniform_distribution_t = typename uniform_distribution_impl<T>::type;

namespace detail {

/**
 * @brief Returns an incrementing seed value for use with UniformRandomGenerator.
 *
 *  The intent behind this is to handle the following case:
 *
 * auto lhs = make_random_wrapped_column<TypeLhs>(10000);
 * auto rhs = make_random_wrapped_column<TypeRhs>(10000);
 *
 * Previously, the binops test framework had a persistent UniformRandomGenerator
 * that would produce unique values across two calls to make_random_wrapped_column()
 * like this.  However that code has been changed and each call to make_random_wrapped_column()
 * now uses a local UniformRandomGenerator object.  If we didn't generate an incrementing seed
 * for each one, every call to make_random_wrapped_column() would return the same values. This
 * fixes that case and also leaves results across multiple test runs deterministic.
 */
uint64_t random_generator_incrementing_seed();

}  // namespace detail

/**
 * @brief Provides uniform random number generation.
 *
 * It is often useful in testing to have a convenient source of random numbers.
 * This class is intended to serve as a base class for test fixtures to provide
 * random number generation. `UniformRandomGenerator::generate()` will generate
 * the next random number in the sequence.
 *
 * Example:
 * ```c++
 * UniformRandomGenerator g(0,100);
 * g.generate(); // Returns a random number in the range [0,100]
 * ```
 *
 * @tparam T The type of values that will be generated.
 */
template <typename T = cudf::size_type, typename Engine = std::default_random_engine>
class UniformRandomGenerator {
 public:
  using uniform_distribution = uniform_distribution_t<T>;  ///< The uniform distribution type for T.

  UniformRandomGenerator() : rng{std::mt19937_64{detail::random_generator_incrementing_seed()}()} {}

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   * @param seed  seed to initialize generator with
   */
  template <typename TL                                                          = T,
            std::enable_if_t<cudf::is_numeric<TL>() && !cudf::is_boolean<TL>()>* = nullptr>
  UniformRandomGenerator(T lower,
                         T upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly random booleans
   *
   * @param lower ignored
   * @param upper ignored
   * @param seed  seed to initialize generator with
   */
  template <typename TL = T, std::enable_if_t<cudf::is_boolean<TL>()>* = nullptr>
  UniformRandomGenerator(T lower,
                         T upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{0.5}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   * @param seed  seed to initialize generator with
   */
  template <typename TL                                                            = T,
            std::enable_if_t<cudf::is_chrono<TL>() or cudf::is_fixed_point<TL>()>* = nullptr>
  UniformRandomGenerator(typename TL::rep lower,
                         typename TL::rep upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Returns the next random number.
   *
   * @return generated random number
   */
  template <typename TL = T, std::enable_if_t<!cudf::is_timestamp<TL>()>* = nullptr>
  T generate()
  {
    return T{dist(rng)};
  }

  /**
   * @brief Returns the next random number.
   * @return generated random number
   */
  template <typename TL = T, std::enable_if_t<cudf::is_timestamp<TL>()>* = nullptr>
  T generate()
  {
    return T{typename T::duration{dist(rng)}};
  }

 private:
  uniform_distribution dist{};  ///< Distribution
  Engine rng;                   ///< Random generator
};

/**
 * @brief Provides temporary directory for temporary test files.
 *
 * Example:
 * ```c++
 * ::testing::Environment* const temp_env =
 *    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment);
 * ```
 */
class TempDirTestEnvironment : public ::testing::Environment {
  temp_directory const tmpdir{"gtest"};

 public:
  /**
   * @brief Get directory path to use for temporary files
   *
   * @return std::string The temporary directory path
   */
  std::string get_temp_dir() { return tmpdir.path(); }

  /**
   * @brief Get a temporary filepath to use for the specified filename
   *
   * @param filename name of the file to be placed in temporary directory.
   * @return std::string The temporary filepath
   */
  std::string get_temp_filepath(std::string filename) { return tmpdir.path() + filename; }
};

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
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
}  // namespace cudf

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
    const char* env_rmm_mode = std::getenv("GTEST_CUDF_RMM_MODE");  // Overridden by CLI options
    const char* env_stream_mode =
      std::getenv("GTEST_CUDF_STREAM_MODE");  // Overridden by CLI options
    auto default_rmm_mode    = env_rmm_mode ? env_rmm_mode : "pool";
    auto default_stream_mode = env_stream_mode ? env_stream_mode : "default";
    options.allow_unrecognised_options().add_options()(
      "rmm_mode",
      "RMM allocation mode",
      cxxopts::value<std::string>()->default_value(default_rmm_mode));
    options.allow_unrecognised_options().add_options()(
      "stream_mode",
      "Whether to use a non-default stream",
      cxxopts::value<std::string>()->default_value(default_stream_mode));
    return options.parse(argc, argv);
  } catch (const cxxopts::OptionException& e) {
    CUDF_FAIL("Error parsing command line options");
  }
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * Should be included in every test program that uses rmm allocators since
 * it maintains the lifespan of the rmm default memory resource.
 * This `main` function is a wrapper around the google test generated `main`,
 * maintaining the original functionality. In addition, this custom `main`
 * function parses the command line to customize test behavior, like the
 * allocation mode used for creating the default memory resource.
 */
#define CUDF_TEST_PROGRAM_MAIN()                                            \
  int main(int argc, char** argv)                                           \
  {                                                                         \
    ::testing::InitGoogleTest(&argc, argv);                                 \
    auto const cmd_opts = parse_cudf_test_opts(argc, argv);                 \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();           \
    auto resource       = cudf::test::create_memory_resource(rmm_mode);     \
    rmm::mr::set_current_device_resource(resource.get());                   \
                                                                            \
    auto const stream_mode = cmd_opts["stream_mode"].as<std::string>();     \
    rmm::cuda_stream const new_default_stream{};                            \
    if (stream_mode == "custom") {                                          \
      auto adapter = make_stream_checking_resource_adaptor(resource.get()); \
      rmm::mr::set_current_device_resource(&adapter);                       \
    }                                                                       \
                                                                            \
    return RUN_ALL_TESTS();                                                 \
  }
