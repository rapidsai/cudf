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

#include <cudf.h>
#include <table.hpp>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns
 *
 *---------------------------------------------------------------------------**/
class CsvReader::Impl {
public:
private:
  const csv_reader_args args_{};

public:
  /**---------------------------------------------------------------------------*
   * @brief JsonReader constructor; throws if the arguments are not supported
   *---------------------------------------------------------------------------**/
  explicit Impl(csv_reader_args const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the args_ data member
   *
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read();

  auto getArgs() const { return args_; }
};

} // namespace cudf
