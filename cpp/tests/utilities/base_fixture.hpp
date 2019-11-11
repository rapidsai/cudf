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

#include "cudf_gtest.hpp"
#include "legacy/cudf_test_utils.cuh"
#include <cudf/wrappers/bool.hpp>

#include <rmm/mr/default_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

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

  static void SetUpTestCase() { ASSERT_RMM_SUCCEEDED(rmmInitialize(nullptr)); }

  static void TearDownTestCase() { ASSERT_RMM_SUCCEEDED(rmmFinalize()); }
};

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
  using uniform_distribution =
      std::conditional_t<std::is_same<T, bool>::value or
                             std::is_same<T, experimental::bool8>::value,
                         std::bernoulli_distribution,
                         std::conditional_t<std::is_floating_point<T>::value,
                                            std::uniform_real_distribution<T>,
                                            std::uniform_int_distribution<T>>>;

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

}  // namespace test
}  // namespace cudf
