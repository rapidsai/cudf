#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <cuda_runtime.h>
//#include "groupby.h"
#include "hash/groupby_compute_api.h"
#include "hash/aggregation_operations.cuh"

template <template <typename> typename aggregation_operation>
gdf_error gdf_group_by_hash(int ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column)
{

  if(ncols > 1) {
    assert( false && "Can only support a single groupby column at this time.");
  }

  gdf_size_type in_size = in_groupby_columns[0]->size;

  gdf_dtype groupby_column_type = in_groupby_columns[0]->dtype;
  gdf_dtype aggregation_column_type = in_aggregation_column->dtype;

  gdf_size_type out_size{0};

  // This is awful
  switch(groupby_column_type)
  {
    //case GDF_INT8:
    //case GDF_INT16:
    case GDF_INT32:
      {
        using groupby_type = int32_t;

        switch(aggregation_column_type)
        {
          case GDF_INT8:
            {
              using aggregation_type = int8_t;
            }
          case GDF_INT16:
            {
              using aggregation_type = int16_t;
            }
          case GDF_INT32:
            {
              using aggregation_type = int32_t;
            }
          case GDF_INT64:
            {
              using aggregation_type = int64_t;
            }
          case GDF_FLOAT32:
            {
              using aggregation_type = float;
            }
          case GDF_FLOAT64:
            {
              using aggregation_type = double;
            }
          default:
            return GDF_UNSUPPORTED_DTYPE;
        }
      }
    case GDF_INT64: // for demo
      {
        using groupby_type = int64_t;

        switch(aggregation_column_type)
        {

          case GDF_INT64: // for demo
            {
              using aggregation_type = int64_t;

              using op_type = aggregation_operation<aggregation_type>;

              groupby_type * in_group_col = static_cast<groupby_type *>(in_groupby_columns[0]->data);
              aggregation_type * in_agg_col = static_cast<aggregation_type *>(in_aggregation_column->data);
              groupby_type * out_group_col = static_cast<groupby_type *>(out_groupby_columns[0]->data);
              aggregation_type * out_agg_col = static_cast<aggregation_type *>(out_aggregation_column->data);

              if(cudaSuccess != GroupbyHash(in_group_col, in_agg_col, in_size, out_group_col, out_agg_col, &out_size, op_type()))
              {
                return GDF_CUDA_ERROR;
              }

              break;
            }
          case GDF_FLOAT64:// for demo
            {
              using aggregation_type = double;

              using op_type = aggregation_operation<aggregation_type>;

              groupby_type * in_group_col = static_cast<groupby_type *>(in_groupby_columns[0]->data);
              aggregation_type * in_agg_col = static_cast<aggregation_type *>(in_aggregation_column->data);
              groupby_type * out_group_col = static_cast<groupby_type *>(out_groupby_columns[0]->data);
              aggregation_type * out_agg_col = static_cast<aggregation_type *>(out_aggregation_column->data);

              if(cudaSuccess != GroupbyHash(in_group_col, in_agg_col, in_size, out_group_col, out_agg_col, &out_size, op_type()))
              {
                return GDF_CUDA_ERROR;
              }
              break;
            }
          case GDF_FLOAT32:
            {
              using aggregation_type = float;
            }
          case GDF_INT8:
            {
              using aggregation_type = int8_t;
            }
          case GDF_INT16:
            {
              using aggregation_type = int16_t;
            }
          case GDF_INT32:
            {
              using aggregation_type = int32_t;
            }
          default:
            return GDF_UNSUPPORTED_DTYPE;

        }
      }
      //case GDF_FLOAT32:
      //case GDF_FLOAT64:
      //case GDF_DATE32:
      //case GDF_DATE64:
      //case GDF_TIMESTAMP:
    default:
      return GDF_UNSUPPORTED_DTYPE;
  }


}
