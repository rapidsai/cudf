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

namespace cudf {
namespace jit {

/**
 * @brief Get the raw pointer to data in a (mutable_)column_view
 */
const void* get_data_ptr(column_view const& view);

/**
 * @brief Get the raw pointer to data in a scalar
 */
const void* get_data_ptr(scalar const& s);

/**
 * @brief Maps a `cudf::data_type` to the name of its corresponding C++ type
 * 
 * When passed a `cudf::data_type`, returns the `std::string` name of the C++
 * type used to represent the data.
 * 
 * Example:
 * @code
 *   auto d = data_type(type_id::INT32);
 *   auto s = jit::getTypeName(d);
 *   // s == std::string("int32_t")
 * @endcode
 * 
 * @param type The data type
 * @return std::string Name of the data type in string
 */
std::string get_type_name(data_type type);
 
} // namespace jit
} // namespace cudf
