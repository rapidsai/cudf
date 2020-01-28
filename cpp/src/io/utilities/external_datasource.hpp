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

#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>

#include <algorithm>
#include <memory>
#include <string>
#include "datasource.hpp"

namespace cudf {
namespace io {
namespace external {

/**
 * @brief Class for reading from a datasource external to the cuDF codebase.
 **/
class external_datasource : datasource {
 public:

  /**
   * Returns the unique identifier for the external datasource. 
   * This value is used in the python/cython layer to specify
   * which external datasource should be used on invocation.
   */
  virtual std::string libcudf_datasource_identifier() = 0;

  /**
   * @brief Base class destructor
   **/
  virtual ~external_datasource(){};

 public:
  std::string DATASOURCE_ID;  // The unique ID the datasource will be referenced by to directly access it.
  
};

}  // namespace external
}  // namespace io
}  // namespace cudf
