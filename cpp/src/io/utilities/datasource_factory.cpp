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

#include "datasource_factory.hpp"

namespace cudf {
namespace io {

datasource_factory::datasource_factory() {
  std::cout << "Creating datasource_factory instance!!!!!!!" << std::endl;
  read_directory();
  std::cout << "Looping through all of the lib directories found" << std::endl;
  for(int i=0; i < libs.size(); i++){
    std::cout << libs[i] << std::endl;
  }
}

}  // namespace io
}  // namespace cudf
