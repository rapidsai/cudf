/**
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

#include <dirent.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>
#include <cudf/utilities/error.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "external_datasource.hpp"

namespace cudf {
namespace io {
namespace external {

/**
 * @brief Factory class for creating and managing instances of external datasources
 */
class datasource_factory {
 public:
  /**
   * Load all of the .so files that are possible candidates for housing external datasources.
   */
  datasource_factory(std::string external_lib_dir);

  /**
   * @brief Base class destructor
   */
  virtual ~datasource_factory(){};

 public:
  /**
   * @brief Get the string path to libcudf External Datasource libraries.
   *
   * This path can be overridden at runtime by defining an environment variable
   * named `EXTERNAL_DATASOURCE_LIB_PATH`. The value of this variable must be a path
   * under which the owming process's user has read privileges.
   *
   * This function returns a path to the cache directory, creating it if it
   * doesn't exist.
   *
   * The default cache directory `$CONDA_PREFIX/lib/external`.
   */
  boost::filesystem::path getExternalLibDir()
  {
    // python user supplied `external_lib_dir_` always has the most precedence
    if (!boost::filesystem::exists(external_lib_dir_)) {
      // Since the python external dir was not supplied check for environment variable.
      auto external_io_lib_path_env = std::getenv("EXTERNAL_DATASOURCE_LIB_PATH");
      if (external_io_lib_path_env != nullptr &&
          boost::filesystem::exists(external_io_lib_path_env)) {
        return boost::filesystem::path(external_io_lib_path_env);
      } else {
        CUDF_EXPECTS(
          external_io_lib_path_env == nullptr,
          "`EXTERNAL_DATASOURCE_LIB_PATH` was set but does not exist on the filesystem.");
        auto conda_prefix = std::getenv("CONDA_PREFIX");
        if (boost::filesystem::exists(conda_prefix)) {
          std::string conda_str = conda_prefix;
          conda_str.append("/lib/external");
          boost::filesystem::path conda_path(conda_str);
          return conda_path;
        } else {
          CUDF_FAIL(
            "`EXTERNAL_DATASOURCE_LIB_PATH` was not specified. External datasources could not be "
            "loaded.");
        }
      }
    } else {
      return external_lib_dir_;
    }
  }

  /**
   * Takes the python/user supplied `external_datasource_id` and returns the external_datasource
   *object to the calling function.
   **/
  external_datasource* external_datasource_by_id(
    std::string unique_id, std::map<std::string, std::string> datasource_confs)
  {
    std::map<std::string, external_datasource_wrapper>::iterator it;
    it = external_libs_.find(unique_id);
    if (it != external_libs_.end()) {
      return it->second.get_external_datasource();
    } else {
      CUDF_FAIL("Unable to find External Datasource specified");
    }
  }

 public:
  /**
   * Wrapper for the external datasource to assist with more concise life cycle management
   */
  class external_datasource_wrapper {
   public:
    external_datasource_wrapper(std::string external_datasource_lib)
    {
      external_datasource_lib_ = external_datasource_lib;
      printf("External Datasource Lib: %s\n", external_datasource_lib_.c_str());

      dl_handle = dlopen(external_datasource_lib_.c_str(), RTLD_LAZY | RTLD_GLOBAL);
      if (!dl_handle) { CUDF_FAIL(dlerror()); }

      ex_ds_load ex_ds = (ex_ds_load)dlsym(dl_handle, "libcudf_external_datasource_load");
      ex_ds_load_from_conf ex_ds_conf =
        (ex_ds_load_from_conf)dlsym(dl_handle, "libcudf_external_datasource_load_from_conf");
      ex_ds_destroy ex_ds_dest =
        (ex_ds_destroy)dlsym(dl_handle, "libcudf_external_datasource_destroy");

      if ((error = dlerror()) != NULL) { CUDF_FAIL(error); }

      ex_ds_        = ex_ds();  // Create external_datasource object
      ds_unique_id_ = ex_ds_->libcudf_datasource_identifier();
      printf("Datasource Unique Identifier: %s\n", ds_unique_id_.c_str());

      // Pending no errors consider the handle alive and open
      open_ = true;
    };

    bool isOpen() { return open_; }

    bool configure(std::map<std::string, std::string> datasource_confs,
                   std::vector<std::string> topics,
                   std::vector<int> partitions)
    {
      datasource_confs_ = datasource_confs;
      return ex_ds_->configure_datasource(datasource_confs);
    }

    std::string unique_id() { return ds_unique_id_; }

    external_datasource* get_external_datasource() { return ex_ds_; }

   private:
    // Shared Object specific variables
    void* dl_handle;
    typedef external_datasource* (*ex_ds_load)();
    typedef external_datasource* (*ex_ds_load_from_conf)(std::map<std::string, std::string>);
    typedef void (*ex_ds_destroy)(external_datasource*);
    char* error;

    std::string ds_unique_id_;
    std::string external_datasource_lib_;
    cudf::io::external::external_datasource* ex_ds_;
    std::map<std::string, std::string> datasource_confs_;
    bool configured_ = false;
    bool open_       = false;
  };

 private:
  void load_external_libs()
  {
    boost::filesystem::path ext_path = getExternalLibDir();
    if (boost::filesystem::exists(ext_path) && boost::filesystem::is_directory(ext_path)) {
      boost::filesystem::directory_iterator it{ext_path};
      boost::filesystem::directory_iterator endit;
      while (it != endit) {
        if (it->path().extension() == EXTERNAL_LIB_SUFFIX) {
          external_datasource_wrapper wrapper(it->path().c_str());
          external_libs_.insert(
            std::pair<std::string, external_datasource_wrapper>(wrapper.unique_id(), wrapper));
        }
        ++it;
      }
    } else {
      CUDF_FAIL("External Datasource directory does not exist");
    }
  }

 private:
  boost::filesystem::path external_lib_dir_;
  std::string EXTERNAL_LIB_SUFFIX = ".so";  // Currently only support .so files.
  std::map<std::string, external_datasource_wrapper> external_libs_;
};

}  // namespace external
}  // namespace io
}  // namespace cudf
