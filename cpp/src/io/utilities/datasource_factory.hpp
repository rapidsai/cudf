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

#include <string>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>

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
  void read_directory() {
    DIR* dirp = opendir(EXTERNAL_LIB_DIR.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        libs.push_back(dp->d_name);
    }
    closedir(dirp);
  }

 private:
  std::string EXTERNAL_LIB_DIR = "/home/jdyer/Development/cudf/external/build";
  std::vector<std::string> libs;
  std::vector<cudf::io::external_datasource> dss;
};

}  // namespace io
}  // namespace cudf
