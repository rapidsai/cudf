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

#include <cudf/column/column_view.hpp>
#include <cudf/legacy/interop.hpp>

namespace cudf {
namespace legacy {

data_type gdf_dtype_to_data_type(gdf_dtype dtype) {}

gdf_dtype data_type_to_gdf_dtype(data_type type) {}

column_view gdf_column_to_view(gdf_column const& col) {}

mutable_column_view gdf_column_to_mutable_view(gdf_column* col) {}

gdf_column view_to_gdf_column(column_view view) {}
}  // namespace legacy
}  // namespace cudf