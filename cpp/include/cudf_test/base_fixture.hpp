/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/file_utilities.hpp>

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/device_memory_resource.hpp>

namespace CUDF_EXPORT cudf {
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
  rmm::device_async_resource_ref _mr{cudf::get_current_device_resource_ref()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   * @return pointer to memory resource
   */
  rmm::device_async_resource_ref mr() { return _mr; }
};

/**
 * @brief Base test fixture that takes a parameter.
 *
 * Example:
 * ```
 * class MyIntTestFixture : public cudf::test::BaseFixtureWithParam<int> {};
 * ```
 */
template <typename T>
class BaseFixtureWithParam : public ::testing::TestWithParam<T> {
  rmm::device_async_resource_ref _mr{cudf::get_current_device_resource_ref()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   * @return pointer to memory resource
   */
  [[nodiscard]] rmm::device_async_resource_ref mr() const { return _mr; }
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

}  // namespace test
}  // namespace CUDF_EXPORT cudf
