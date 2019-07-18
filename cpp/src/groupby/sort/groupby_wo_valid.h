/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Alexander Ocsa <alexander@blazingdb.com>
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

#include <cudf/cudf.h>
#include "rmm/thrust_rmm_allocator.h"

gdf_error gdf_group_by_sort_wo_valids(
    gdf_size_type ncols, gdf_column *in_groupby_columns[],
    gdf_column *in_aggregation_column, gdf_column *out_groupby_columns[],
    gdf_column *out_aggregation_column, gdf_agg_op agg_op, gdf_context *ctxt,
    rmm::device_vector<int32_t> &sorted_indices);