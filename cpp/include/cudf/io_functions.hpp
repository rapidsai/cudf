/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <memory>

#include "cudf.h"
#include <cudf/legacy/table.hpp>

namespace cudf {

/*
 * @brief Interface to parse Avro data to cuDF columns.
 *
 * @param[in] args Arguments for controlling reading behavior
 *
 * @return cudf::table Object that contains the array of gdf_columns
 */
cudf::table read_avro(avro_read_arg const &args);

/*
 * @brief Interface to parse CSV data to cuDF columns.
 *
 * @param[in] args Arguments for controlling reading behavior
 *
 * @return cudf::table Object that contains the array of gdf_columns
 */
cudf::table read_csv(csv_read_arg const &args);

/*
 * @brief Interface to parse JSON data to cuDF columns
 *
 * @param[in] args Arguments for controlling reading behavior
 *
 * @return cudf::table Object that contains the array of gdf_columns
 */
cudf::table read_json(json_read_arg const &args);

/*
 * @brief Interface to parse ORC data to cuDF columns
 *
 * @param[in] args Arguments for controlling reading behavior
 *
 * @return cudf::table Object that contains the array of gdf_columns
 */
cudf::table read_orc(orc_read_arg const &args);

/*
 * @brief Interface to output cuDF columns to ORC format
 *
 * @param[in] args Arguments for controlling writing behavior
 */
void write_orc(orc_write_arg const &args);

/*
 * @brief Interface to parse Parquet data to cuDF columns.
 *
 * @param[in] args Arguments for controlling reading behavior
 *
 * @return cudf::table Object that contains the array of gdf_columns
 */
cudf::table read_parquet(parquet_read_arg const &args);

}  // namespace cudf
