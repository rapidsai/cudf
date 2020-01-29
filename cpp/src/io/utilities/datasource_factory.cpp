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
#include <string>
#include <map>

namespace cudf {
namespace io {
namespace external {

datasource_factory::datasource_factory(std::string external_lib_dir) : EXTERNAL_LIB_DIR(external_lib_dir) {
  load_external_libs();
}

}  // namespace external
}  // namespace io
}  // namespace cudf
