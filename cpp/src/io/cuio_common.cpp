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

#include "cuio_common.hpp"

gdf_dtype convertStringToDtype(const std::string &dtype) {
  if (dtype == "str")
    return GDF_STRING;
  if (dtype == "date")
    return GDF_DATE64;
  if (dtype == "date32")
    return GDF_DATE32;
  if (dtype == "date64")
    return GDF_DATE64;
  if (dtype == "timestamp")
    return GDF_TIMESTAMP;
  if (dtype == "category")
    return GDF_CATEGORY;
  if (dtype == "float")
    return GDF_FLOAT32;
  if (dtype == "float32")
    return GDF_FLOAT32;
  if (dtype == "float64")
    return GDF_FLOAT64;
  if (dtype == "double")
    return GDF_FLOAT64;
  if (dtype == "short")
    return GDF_INT16;
  if (dtype == "int")
    return GDF_INT32;
  if (dtype == "int32")
    return GDF_INT32;
  if (dtype == "int64")
    return GDF_INT64;
  if (dtype == "long")
    return GDF_INT64;
  return GDF_invalid;
}
