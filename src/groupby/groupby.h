#include <gdf/gdf.h>

gdf_error gdf_group_by_hash(int ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column);
//                            gdf_agg_op op);
