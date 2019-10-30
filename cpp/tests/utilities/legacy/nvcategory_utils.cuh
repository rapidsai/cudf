/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Felipe Aramburu <felipe@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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


#ifndef CUDF_NVCATEGORY_UTILS_CUH_
#define CUDF_NVCATEGORY_UTILS_CUH_

#include <cudf/types.h>
#include <vector>
#include <tuple>
#include <string>

namespace cudf {
namespace test {

std::string random_string(size_t len = 15, std::string const &allowed_chars = "abcdefghijklmnaoqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");

gdf_column * create_nv_category_column(cudf::size_type num_rows, bool repeat_strings);

gdf_column * create_nv_category_column_strings(const char ** string_host_data, cudf::size_type num_rows);

const char ** generate_string_data(cudf::size_type num_rows, size_t length, bool print=false);

std::tuple<std::vector<std::string>, std::vector<cudf::valid_type>> nvcategory_column_to_host(gdf_column * column);

} // namespace test
} // namespace cudf

#endif