/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

std::string type_to_name(data_type type) { return type_dispatcher(type, type_to_name_impl{}); }

std::string device_storage_type_name(data_type type)
{
  switch (type.id()) {
    case cudf::type_id::DECIMAL32: return "int32_t";
    case cudf::type_id::DECIMAL64: return "int64_t";
    case cudf::type_id::DECIMAL128: return "__int128_t";
    default: return cudf::type_to_name(type);
  };
}

}  // namespace cudf
