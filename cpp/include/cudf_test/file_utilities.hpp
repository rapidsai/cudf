/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

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
