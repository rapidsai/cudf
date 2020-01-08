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

#include <cudf/types.hpp>

namespace cudf {
namespace io {
namespace json {
namespace gpu {

void convertJsonToColumns(rmm::device_buffer const& input_data, 
                          data_type *const dtypes, void *const *output_columns,
                          cudf::size_type num_records, cudf::size_type num_columns,  
                          const uint64_t *rec_starts,                        
                          bitmask_type *const *valid_fields, cudf::size_type *num_valid_fields,
                          ParseOptions const& opts);

void DetectDataTypes(ColumnInfo *column_infos,
                     const char *data, size_t data_size, 
                     const ParseOptions &options, int num_columns,
                     const uint64_t *rec_starts, cudf::size_type num_records);

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
