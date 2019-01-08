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

#include <thrust/scatter.h>
#include "copying.hpp"
#include "cudf.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/type_dispatcher.hpp"

namespace cudf {

namespace detail {
gdf_error scatter(table const* source_table, gdf_index_type const scatter_map[],
                  table* destination_table, cudaStream_t stream = 0) {
  return GDF_SUCCESS;
}
}  // namespace detail

gdf_error scatter(table const* source_table, gdf_index_type const scatter_map[],
                  table* destination_table) {
  return detail::scatter(source_table, scatter_map, destination_table);
}
}  // namespace cudf