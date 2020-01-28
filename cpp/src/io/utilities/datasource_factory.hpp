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

#include <string>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <dlfcn.h>

#include "external_datasource.hpp"

namespace cudf {
namespace io {

/**
 * @brief Factory class for creating and managing instances of external datasources
 **/
class datasource_factory {
 public:

  /**
   * Load all of the .so files that are possible candidates for housing external datasources.
   */
  datasource_factory();

  /**
   * @brief Base class destructor
   **/
  virtual ~datasource_factory(){};
 
 private:
  void list_external_libs() {
    DIR* dirp = opendir(EXTERNAL_LIB_DIR.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
      if (has_suffix(dp->d_name, EXTERNAL_LIB_SUFFIX)) {
        std::string filename(EXTERNAL_LIB_DIR);
        filename.append("/");
        filename.append(dp->d_name);
        libs.push_back(filename);
      }
    }
    closedir(dirp);
  }

  bool has_suffix(const std::string &str, const std::string &suffix)
  {
    return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

 private:
  std::string EXTERNAL_LIB_DIR = "/home/jdyer/Development/cudf/external/build";
  std::string EXTERNAL_LIB_SUFFIX = ".so";     // Currently only support .so files.
  std::vector<std::string> libs;
  std::vector<cudf::io::external_datasource> dss;
};

}  // namespace io
}  // namespace cudf
