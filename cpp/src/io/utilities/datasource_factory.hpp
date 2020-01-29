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
#include <map>
#include <dlfcn.h>

#include "external_datasource.hpp"

namespace cudf {
namespace io {
namespace external {

/**
 * @brief Factory class for creating and managing instances of external datasources
 **/
class datasource_factory {
 public:

  /**
   * Load all of the .so files that are possible candidates for housing external datasources.
   */
  datasource_factory(std::string external_lib_dir);

  /**
   * @brief Base class destructor
   **/
  virtual ~datasource_factory(){};
 
 public:
  
  external_datasource* external_datasource_by_id(std::string unique_id, std::map<std::string, std::string> datasource_confs) {
    std::map<std::string, external_datasource_wrapper>::iterator it;
    it = external_libs_.find(unique_id);
    if (it != external_libs_.end()) {
      return it->second.get_external_datasource();
    } else {
      printf("Unable to find External Datasource with Identifier: '%s'\n", unique_id.c_str());
      //TODO: What exactly should we do here???
      exit(1);
    }
  }

 public:
  
  /**
   * Wrapper for the external datasource to assist with more concise life cycle management
   */
  class external_datasource_wrapper {
    public:
      external_datasource_wrapper(std::string external_datasource_lib){
        external_datasource_lib_ = external_datasource_lib;

        dl_handle = dlopen(external_datasource_lib_.c_str(), RTLD_LAZY);
        //TODO: Explore best approach for error handling here that fits into the cuDF paradigm
        if (!dl_handle) {
          fputs (dlerror(), stderr);
          exit(1);
        }

        ex_ds_load ex_ds = (ex_ds_load) dlsym(dl_handle, "libcudf_external_datasource_load");
        ex_ds_load_from_conf ex_ds_conf = (ex_ds_load_from_conf) dlsym(dl_handle, "libcudf_external_datasource_load_from_conf");
        ex_ds_destroy ex_ds_dest = (ex_ds_destroy) dlsym(dl_handle, "libcudf_external_datasource_destroy");

        //TODO: Explore best approach for error handling here that fits into the cuDF paradigm
        if ((error = dlerror()) != NULL)  {
          fputs(error, stderr);
          exit(1);
        }

        ex_ds_ = ex_ds();
        ds_unique_id_ = ex_ds_->libcudf_datasource_identifier();

        // Pending no errors consider the handle alive and open
        open_ = true;
      };

      bool isOpen() {
        return open_;
      }

      std::string unique_id() {
        return ds_unique_id_;
      }

      external_datasource* get_external_datasource() {
        return ex_ds_;
      }

    private:
      // Shared Object specific variables
      void *dl_handle;
      typedef external_datasource* (*ex_ds_load)();
      typedef external_datasource* (*ex_ds_load_from_conf)(std::map<std::string, std::string>);
      typedef void (*ex_ds_destroy) (external_datasource*);
      char *error;

      std::string ds_unique_id_;
      std::string external_datasource_lib_;
      cudf::io::external::external_datasource *ex_ds_;
      bool open_ = false;
  };

 private:
  void load_external_libs() {
    DIR* dirp = opendir(EXTERNAL_LIB_DIR.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
      if (has_suffix(dp->d_name, EXTERNAL_LIB_SUFFIX)) {
        std::string filename(EXTERNAL_LIB_DIR);
        filename.append("/");
        filename.append(dp->d_name);

        // Load and wrap the external datasource.
        external_datasource_wrapper wrapper(filename);
        external_libs_.insert(std::pair<std::string, external_datasource_wrapper>(wrapper.unique_id(), wrapper));

        // Print out information for user
        printf("External Datasource: '%s' loaded from shared object: '%s'\n", wrapper.unique_id().c_str(), filename.c_str());
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
  std::string EXTERNAL_LIB_DIR;
  std::string EXTERNAL_LIB_SUFFIX = ".so";     // Currently only support .so files.
  std::map<std::string, external_datasource_wrapper> external_libs_;
};

}  // namespace external
}  // namespace io
}  // namespace cudf
