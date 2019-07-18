
#pragma once

#include "rmm/thrust_rmm_allocator.h"
#include <cudf/cudf.h>

gdf_error gdf_group_by_count_wo_valids(
    gdf_size_type ncols, gdf_column *in_groupby_columns[],
    gdf_column *in_aggregation_column, gdf_column *out_groupby_columns[],
    gdf_column *out_aggregation_column, gdf_agg_op agg_op, gdf_context *ctxt,
    rmm::device_vector<int32_t> &sorted_indices);
