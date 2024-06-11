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

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/error.hpp>

#include <ftw.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

/**
 * @brief RAII class for creating a temporary directory.
 *
 */
class CUDF_EXPORT temp_directory {
  std::string _path;

 public:
  /**
   * @brief Construct a new temp directory object
   *
   * @param base_name The base name of the temporary directory
   */
  temp_directory(std::string const& base_name)
  {
    std::string dir_template{std::filesystem::temp_directory_path().string()};

    dir_template += "/" + base_name + ".XXXXXX";
    auto const tmpdirptr = mkdtemp(const_cast<char*>(dir_template.data()));
    CUDF_EXPECTS(tmpdirptr != nullptr, "Temporary directory creation failure: " + dir_template);

    _path = dir_template + "/";
  }

  temp_directory& operator=(temp_directory const&) = delete;
  temp_directory(temp_directory const&)            = delete;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object
   */
  temp_directory& operator=(temp_directory&&) = default;
  temp_directory(temp_directory&&)            = default;  ///< Move constructor

  ~temp_directory() { std::filesystem::remove_all(std::filesystem::path{_path}); }

  /**
   * @brief Returns the path of the temporary directory
   *
   * @return string path of the temporary directory
   */
  [[nodiscard]] std::string const& path() const { return _path; }
};
