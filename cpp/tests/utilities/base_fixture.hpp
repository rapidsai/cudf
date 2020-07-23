/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/utilities/traits.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/cxxopts.hpp>

#include <rmm/mr/device/cnmem_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <ftw.h>
#include <random>

namespace cudf {
namespace test {
/**
 * @brief Base test fixture class from which all libcudf tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cudf::test::BaseFixture {};
 * ```
 **/
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource *_mr{rmm::mr::get_default_resource()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   **/
  rmm::mr::device_memory_resource *mr() { return _mr; }
};

template <typename T, typename Enable = void>
struct uniform_distribution_impl {
};
template <typename T>
struct uniform_distribution_impl<
  T,
  std::enable_if_t<std::is_integral<T>::value && not cudf::is_boolean<T>()>> {
  using type = std::uniform_int_distribution<T>;
};

template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_floating_point<T>::value>> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<cudf::is_boolean<T>()>> {
  using type = std::bernoulli_distribution;
};

template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<cudf::is_chrono<T>()>> {
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
 *
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
 **/
template <typename T = cudf::size_type, typename Engine = std::default_random_engine>
class UniformRandomGenerator {
 public:
  using uniform_distribution = uniform_distribution_t<T>;

  UniformRandomGenerator() : rng{std::mt19937_64{detail::random_generator_incrementing_seed()}()} {}

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   */
  template <typename TL = T, std::enable_if_t<!cudf::is_chrono<TL>()> * = nullptr>
  UniformRandomGenerator(T lower,
                         T upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   */
  template <typename TL = T, std::enable_if_t<cudf::is_chrono<TL>()> * = nullptr>
  UniformRandomGenerator(typename TL::rep lower,
                         typename TL::rep upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Returns the next random number.
   **/
  T generate() { return T{dist(rng)}; }

 private:
  uniform_distribution dist{};  ///< Distribution
  Engine rng;                   ///< Random generator
};

class temp_directory {
  std::string _path;

 public:
  temp_directory(const std::string &base_name)
  {
    std::string dir_template("/tmp");
    if (const char *env_p = std::getenv("WORKSPACE")) dir_template = env_p;
    dir_template += "/" + base_name + ".XXXXXX";
    auto const tmpdirptr = mkdtemp(const_cast<char *>(dir_template.data()));
    if (tmpdirptr == nullptr) CUDF_FAIL("Temporary directory creation failure: " + dir_template);
    _path = dir_template + "/";
  }

  static int rm_files(const char *pathname, const struct stat *sbuf, int type, struct FTW *ftwb)
  {
    return remove(pathname);
  }

  ~temp_directory()
  {
    // TODO: should use std::filesystem instead, once C++17 support added
    nftw(_path.c_str(), rm_files, 10, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
  }

  const std::string &path() const { return _path; }
};

/**
 * @brief Provides temporary directory for temporary test files.
 *
 * Example:
 * ```c++
 * ::testing::Environment* const temp_env =
 *    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment);
 * ```
 **/
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
   * @return std::string The temporary filepath
   */
  std::string get_temp_filepath(std::string filename) { return tmpdir.path() + filename; }
};

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
inline std::unique_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const &allocation_mode)
{
  if (allocation_mode == "cuda") return std::make_unique<rmm::mr::cuda_memory_resource>();
  if (allocation_mode == "pool") return std::make_unique<rmm::mr::cnmem_memory_resource>();
  if (allocation_mode == "managed") return std::make_unique<rmm::mr::managed_memory_resource>();
  CUDF_FAIL("Invalid RMM allocation mode: " + allocation_mode);
}

}  // namespace test
}  // namespace cudf

/**
 * @brief Parses the cuDF test command line options.
 *
 * Currently only supports 'rmm_mode' string paramater, which set the rmm
 * allocation mode. The default value of the parameter is 'pool'.
 *
 * @return Parsing results in the form of unordered map
 */
inline auto parse_cudf_test_opts(int argc, char **argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuDF tests command line options");
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"));

    return options.parse(argc, argv);
  } catch (const cxxopts::OptionException &e) {
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
 *
 */
#define CUDF_TEST_PROGRAM_MAIN()                                        \
  int main(int argc, char **argv)                                       \
  {                                                                     \
    ::testing::InitGoogleTest(&argc, argv);                             \
    auto const cmd_opts = parse_cudf_test_opts(argc, argv);             \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();       \
    auto resource       = cudf::test::create_memory_resource(rmm_mode); \
    rmm::mr::set_default_resource(resource.get());                      \
    return RUN_ALL_TESTS();                                             \
  }
