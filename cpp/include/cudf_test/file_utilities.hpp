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

#include <cstdio>
#include <cstdlib>
#include <string>

#include <ftw.h>

#include <cudf/utilities/error.hpp>

class temp_directory {
  std::string _path;

 public:
  temp_directory(const std::string& base_name)
  {
    std::string dir_template("/tmp");
    if (const char* env_p = std::getenv("WORKSPACE")) dir_template = env_p;
    dir_template += "/" + base_name + ".XXXXXX";
    auto const tmpdirptr = mkdtemp(const_cast<char*>(dir_template.data()));
    if (tmpdirptr == nullptr) CUDF_FAIL("Temporary directory creation failure: " + dir_template);
    _path = dir_template + "/";
  }

  static int rm_files(const char* pathname, const struct stat* sbuf, int type, struct FTW* ftwb)
  {
    return std::remove(pathname);
  }

  ~temp_directory()
  {
    // TODO: should use std::filesystem instead, once C++17 support added
    nftw(_path.c_str(), rm_files, 10, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
  }

  const std::string& path() const { return _path; }
};
