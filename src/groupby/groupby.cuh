#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <cuda_runtime.h>
//#include "groupby.h"
#include "hash/groupby_compute_api.h"
#include "hash/aggregation_operations.cuh"

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This function provides the libgdf entry point for a hash-based group-by.
 * Performs a Group-By operation on an arbitrary number of columns with a single aggregation column.
 * 
 * @Param[in] ncols The number of columns to group-by
 * @Param[in] in_groupby_columns[] The columns to group-by
 * @Param[in,out] in_aggregation_column The column to perform the aggregation on
 * @Param[in,out] out_groupby_columns[] A preallocated buffer to store the resultant group-by columns
 * @Param[in,out] out_aggregation_column A preallocated buffer to store the resultant aggregation column
 * @tparam[in] aggregation_operation A functor that defines the aggregation operation
 * 
 * @Returns gdf_error
 */
/* ----------------------------------------------------------------------------*/
template <template <typename> typename aggregation_operation>
gdf_error gdf_group_by_hash(int ncols,               
                            gdf_column* in_groupby_columns[],        
                            gdf_column* in_aggregation_column,       
                            gdf_column* out_groupby_columns[],
                            gdf_column* out_aggregation_column,
                            bool sort_result = false)
{

  // TODO Currently only supports a single groupby column
  if(ncols > 1) {
    assert( false && "Can only support a single groupby column at this time.");
  }

  gdf_size_type in_size = in_groupby_columns[0]->size;
  gdf_dtype groupby_column_type = in_groupby_columns[0]->dtype;
  gdf_dtype aggregation_column_type = in_aggregation_column->dtype;

  // The size of the result column(s)
  gdf_size_type out_size{0};

  // TODO Currently only supports GDF_INT64 groupby columns and GDF_INT64, GDF_FLOAT64 aggregation columns
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
        break;
      }
    case GDF_INT64: // for demo
      {
        using groupby_type = int64_t;

        switch(aggregation_column_type)
        {
          case GDF_INT64: // for demo
            {
              using aggregation_type = int64_t;

              // Template the functor on the type of the aggregation column
              using op_type = aggregation_operation<aggregation_type>;

              // Cast the void* data to the appropriate type
              groupby_type * in_group_col = static_cast<groupby_type *>(in_groupby_columns[0]->data);
              aggregation_type * in_agg_col = static_cast<aggregation_type *>(in_aggregation_column->data);
              groupby_type * out_group_col = static_cast<groupby_type *>(out_groupby_columns[0]->data);
              aggregation_type * out_agg_col = static_cast<aggregation_type *>(out_aggregation_column->data);

              if(cudaSuccess != GroupbyHash(in_group_col, in_agg_col, in_size, out_group_col, out_agg_col, &out_size, op_type(), sort_result))
              {
                return GDF_CUDA_ERROR;
              }

              // Update the size of the result
              out_groupby_columns[0]->size = out_size;
              out_aggregation_column->size = out_size;

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

              if(cudaSuccess != GroupbyHash(in_group_col, in_agg_col, in_size, out_group_col, out_agg_col, &out_size, op_type(), sort_result))
              {
                return GDF_CUDA_ERROR;
              }

              out_groupby_columns[0]->size = out_size;
              out_aggregation_column->size = out_size;

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
            std::cout << "Unsupported aggregation column type: " << aggregation_column_type << std::endl;
            return GDF_UNSUPPORTED_DTYPE;

        }
        break;

      }
      //case GDF_FLOAT32:
      //case GDF_FLOAT64:
      //case GDF_DATE32:
      //case GDF_DATE64:
      //case GDF_TIMESTAMP:
    default:
      std::cout << "Unsupported groupby column type:" << groupby_column_type << std::endl;
      return GDF_UNSUPPORTED_DTYPE;
  }

  return GDF_SUCCESS;

}
