#include "cudf.h"
#include "new_groupby.hpp"

gdf_error gdf_group_by_sum(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  gdf_agg_op op{GDF_SUM};
  return gdf_group_by(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);
}

gdf_error gdf_group_by_min(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  gdf_agg_op op{GDF_MIN};
  return gdf_group_by(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);
}

gdf_error gdf_group_by_max(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  gdf_agg_op op{GDF_MAX};
  return gdf_group_by(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);

}

gdf_error gdf_group_by_avg(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  gdf_agg_op op{GDF_AVG};
  return gdf_group_by(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);

}

gdf_error gdf_group_by_count(int ncols,                    // # columns
                             gdf_column** cols,            //input cols
                             gdf_column* col_agg,          //column to aggregate on
                             gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                             gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                             gdf_column* out_col_agg,      //aggregation result
                             gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{
  gdf_agg_op op{GDF_COUNT};
  return gdf_group_by(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);
}

gdf_error gdf_group_by_count_distinct(int ncols,                    // # columns
                           gdf_column** cols,            //input cols with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
                           gdf_column* col_agg,          //column to aggregate on with 0 null_count otherwise GDF_VALIDITY_UNSUPPORTED is returned
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{
  gdf_agg_op op{GDF_COUNT_DISTINCT};
  return gdf_group_by_sort(cols,
                      ncols,
                      &col_agg,
                      1,
                      &op,
                      out_col_values,
                      &out_col_agg,
                      ctxt);
}
