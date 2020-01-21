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

#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/wrappers/bool.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <ftw.h>
#include <random>

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Base test fixture class from which all libcudf tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cudf::test::BaseFixture {};
 * ```
 *---------------------------------------------------------------------------**/
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource* _mr{rmm::mr::get_default_resource()};

 public:
  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheritng from this fixture
   *---------------------------------------------------------------------------**/
  rmm::mr::device_memory_resource* mr() { return _mr; }

  static void SetUpTestCase() { ASSERT_EQ(rmmInitialize(nullptr), RMM_SUCCESS); }

  static void TearDownTestCase() { ASSERT_EQ(rmmFinalize(), RMM_SUCCESS); }
};

template <typename T, typename Enable = void>
struct uniform_distribution_impl{};
template<typename T>
struct uniform_distribution_impl<T,
  std::enable_if_t<std::is_integral<T>::value && not cudf::is_boolean<T>()> > {
   using type = std::uniform_int_distribution<T>;
};

template<typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_floating_point<T>::value > > {
   using type = std::uniform_real_distribution<T>;
};

template<typename T>
struct uniform_distribution_impl<T, std::enable_if_t<cudf::is_boolean<T>() > > {
   using type = std::bernoulli_distribution;
};

template<typename T>
struct uniform_distribution_impl<T, std::enable_if_t<cudf::is_timestamp<T>() > > {
   using type = std::uniform_int_distribution<typename T::duration::rep>;
};

template <typename T>
using uniform_distribution_t = typename uniform_distribution_impl<T>::type;

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
template <typename T = cudf::size_type,
          typename Engine = std::default_random_engine>
class UniformRandomGenerator {
 public:
  using uniform_distribution = uniform_distribution_t<T>;

  UniformRandomGenerator() = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   *---------------------------------------------------------------------------**/
  UniformRandomGenerator(T lower, T upper) : dist{lower, upper} {}

  /**---------------------------------------------------------------------------*
   * @brief Returns the next random number.
   *---------------------------------------------------------------------------**/
  T generate() { return T{dist(rng)}; }

 private:
  uniform_distribution dist{};         ///< Distribution
  Engine rng{std::random_device{}()};  ///< Random generator
};

/**---------------------------------------------------------------------------*
 * @brief Provides temporary directory for temporary test files.
 *
 * Example:
 * ```c++
 * ::testing::Environment* const temp_env =
 *    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment);
 * ```
 *---------------------------------------------------------------------------**/
class TempDirTestEnvironment : public ::testing::Environment {
 public:
  std::string tmpdir;

  void SetUp() {
    char tmp_format[] = "/tmp/gtest.XXXXXX";
    tmpdir = mkdtemp(tmp_format);
    tmpdir += "/";
  }

  void TearDown() {
    // TODO: should use std::filesystem instead, once C++17 support added
    nftw(tmpdir.c_str(), rm_files, 10, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
  }

  static int rm_files(const char *pathname, const struct stat *sbuf, int type,
                      struct FTW *ftwb) {
    return remove(pathname);
  }

  /**
   * @brief Get directory path to use for temporary files
   *
   * @return std::string The temporary directory path
   */
  std::string get_temp_dir() { return tmpdir; }

  /**
   * @brief Get a temporary filepath to use for the specified filename
   *
   * @return std::string The temporary filepath
   */
  std::string get_temp_filepath(std::string filename) {
    return tmpdir + filename;
  }
};

}  // namespace test
}  // namespace cudf
