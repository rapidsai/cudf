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

#include "datasource_factory.hpp"

namespace cudf {
namespace io {
namespace external {

datasource_factory::datasource_factory() {
  list_external_libs();

  // Open each one of the external libraries and query for its unique identifier and print
  for(int i=0; i < libs.size(); i++) {
    // std::cout << libs[i] << std::endl;
    // void *handle;
    // std::string (*identifier)();
    // char *error;

    // handle = dlopen(libs[i].c_str(), RTLD_LAZY);
    // if (!handle) {
    //   fputs (dlerror(), stderr);
    //   exit(1);
    // }

    // identifier = (std::string *) dlsym(handle, "hello");
    // if ((error = dlerror()) != NULL)  {
    //   fputs(error, stderr);
    //   exit(1);
    // }

    // printf("About to print the identifier .....\n");
    // printf ("%s\n", (*identifier).c_str());
    // dlclose(handle);

    std::cout << libs[i] << std::endl;
    void *handle;
    typedef external_datasource* (*ex_ds_t)();
    char *error;

    handle = dlopen(libs[i].c_str(), RTLD_LAZY);
    if (!handle) {
      fputs (dlerror(), stderr);
      exit(1);
    }

    ex_ds_t ex_ds = (ex_ds_t) dlsym(handle, "libcudf_external_datasource_load");
    if ((error = dlerror()) != NULL)  {
      fputs(error, stderr);
      exit(1);
    }

    printf("About to print the identifier .....\n");
    external_datasource *result = ex_ds();
    printf("Result %s\n", result->libcudf_datasource_identifier().c_str());
    dlclose(handle);
  }
}

}  // namespace external
}  // namespace io
}  // namespace cudf
