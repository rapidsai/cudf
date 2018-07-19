#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include "groupby.h"
#include "hash/groupby_compute_api.h"
#include "hash/aggregation_operations.cuh"

gdf_error gdf_group_by_hash(int ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column) {

  if(ncols > 1) {
    assert( false && "Can only support a single groupby column at this time.");
  }

}
