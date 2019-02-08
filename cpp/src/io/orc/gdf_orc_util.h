/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

// file: gdf_orc_util.h
// gdf interface and utility

#ifndef __GDF_ORC_UTIL_H__
#define __GDF_ORC_UTIL_H__

#include <cudf.h>
#include "orc_types.h"
#include "orc_read.h"

namespace cudf {
namespace orc {

CudaOrcReader* gdf_create_orc_reader();

gdf_error gdf_orc_convertErrorCode(CudaOrcError_t err);

gdf_dtype gdf_orc_convertDataKind(ORCTypeKind kind);

gdf_error gdf_orc_set_column_name(gdf_column* col, const std::string& name);
gdf_error gdf_orc_release_column_name(gdf_column* col);

}   // namespace orc
}   // namespace cudf

#endif //  __GDF_ORC_UTIL_H__
