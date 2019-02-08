#include <cassert>
#include "cudf.h"
#include "new_groupby.hpp"
#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/error_utils.h"
#include "aggregation_operations.hpp"
#include "groupby/hash_groupby.cuh"
#include "string/nvcategory_util.cuh"

namespace{
  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Verifies that a set gdf_columns contain non-null data buffers, and are all 
   * of the same size. 
   *
   *
   * TODO: remove when null support added. 
   *
   * Also ensures that the columns do not contain any null values
   * 
   * @Param[in] first Pointer to first gdf_column in set
   * @Param[in] last Pointer to one past the last column in set
   * 
   * @Returns GDF_DATASET_EMPTY if a column contains a null data buffer, 
   * GDF_COLUMN_SIZE_MISMATCH if the columns are not of equal length, 
   */
  /* ----------------------------------------------------------------------------*/
  gdf_error verify_columns(gdf_column * cols[], int num_cols)
  {
    GDF_REQUIRE((nullptr != cols[0]), GDF_DATASET_EMPTY);

    gdf_size_type const required_size{cols[0]->size};

    for(int i = 0; i < num_cols; ++i)
    {
      GDF_REQUIRE(nullptr != cols[i], GDF_DATASET_EMPTY); 
      GDF_REQUIRE(nullptr != cols[i]->data, GDF_DATASET_EMPTY);
      GDF_REQUIRE(required_size == cols[i]->size, GDF_COLUMN_SIZE_MISMATCH );

      // TODO Remove when null support for hash-based groupby is added
      GDF_REQUIRE(nullptr == cols[i]->valid || 0 == cols[i]->null_count, GDF_VALIDITY_UNSUPPORTED);
    }
    return GDF_SUCCESS;
  }
} // anonymous namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Groupby operation for an arbitrary number of key columns and an
 * arbitrary number of aggregation columns.
 *
 * "Groupby" is a reduce-by-key operation where rows in one or more "key" columns
 * act as the keys and one or more "aggregation" columns hold the values that will
 * be reduced. 
 *
 * The output of the operation is the set of key columns that hold all the unique keys
 * from the input key columns and a set of aggregation columns that hold the specified
 * reduction among all identical keys.
 * 
 * @Param[in] in_key_columns[] The input key columns
 * @Param[in] num_key_columns The number of input columns to groupby
 * @Param[in] in_aggregation_columns[] The columns that will be aggregated
 * @Param[in] num_aggregation_columns The number of columns that will be aggregated
 * @Param[in] agg_ops[] The aggregation operations to perform. The number of aggregation
 * operations must be equal to the number of aggregation columns, such that agg_op[i]
 * will be applied to in_aggregation_columns[i]
 * @Param[in,out] out_key_columns[] Preallocated buffers for the output key columns
 * columns
 * @Param[in,out] out_aggregation_columns[] Preallocated buffers for the output 
 * aggregation columns
 * @Param[in] options Structure that controls behavior of groupby operation, i.e.,
 * sort vs. hash-based implementation, whether or not the output will be sorted,
 * etc. See definition of gdf_context.
 * 
 * @Returns GDF_SUCCESS upon succesful completion. Otherwise appropriate error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_group_by(gdf_column* in_key_columns[],
                       int num_key_columns,
                       gdf_column* in_aggregation_columns[],
                       int num_aggregation_columns,
                       gdf_agg_op agg_ops[],
                       gdf_column* out_key_columns[],
                       gdf_column* out_aggregation_columns[],
                       gdf_context* options)
{

  // TODO: Remove when single pass multi-agg is implemented
  if(num_aggregation_columns > 1)
    assert(false && "Only 1 aggregation column currently supported.");

  // TODO: Remove when the `flag_method` member is removed from `gdf_context`
  if(GDF_SORT == options->flag_method)
    assert(false && "Sort-based groupby is no longer supported.");

  // Ensure inputs aren't null
  if( (0 == num_key_columns)
      || (0 == num_aggregation_columns)
      || (nullptr == in_key_columns)
      || (nullptr == in_aggregation_columns)
      || (nullptr == agg_ops)
      || (nullptr == out_key_columns)
      || (nullptr == out_aggregation_columns)
      || (nullptr == options))
  {
    return GDF_DATASET_EMPTY;
  }

  // Return immediately if inputs are empty
  GDF_REQUIRE(0 != in_key_columns[0]->size, GDF_SUCCESS);
  GDF_REQUIRE(0 != in_aggregation_columns[0]->size, GDF_SUCCESS);

  auto result = verify_columns(in_key_columns, num_key_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  result = verify_columns(in_aggregation_columns, num_aggregation_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_GROUPBY", GROUPBY_COLOR);


  bool sort_result = false;

  if( 0 != options->flag_sort_result){
    sort_result = true;
  }

  //Check that user is not trying to sum or avg string columns
  for(int aggregation_index = 0; aggregation_index < num_aggregation_columns; aggregation_index++){
	  if(( agg_ops[aggregation_index] == GDF_SUM ||
		   agg_ops[aggregation_index] == GDF_AVG ) &&
		   in_aggregation_columns[aggregation_index]->dtype == GDF_STRING_CATEGORY){
		  return GDF_UNSUPPORTED_DTYPE;
	  }

  }

  // TODO: Only a single aggregator supported right now
  gdf_agg_op op{agg_ops[0]};

  switch(op)
  {
    case GDF_MAX:
      {
        gdf_error_code = gdf_group_by_hash<max_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_MIN:
      {
        gdf_error_code = gdf_group_by_hash<min_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_SUM:
      {
        gdf_error_code = gdf_group_by_hash<sum_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_COUNT:
      {
        gdf_error_code = gdf_group_by_hash<count_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   sort_result);
        break;
      }
    case GDF_AVG:
      {
        gdf_error_code = gdf_group_by_hash_avg(num_key_columns,
                                               in_key_columns,
                                               in_aggregation_columns[0],
                                               out_key_columns,
                                               out_aggregation_columns[0]);

        break;
      }
    default:
      std::cerr << "Unsupported aggregation method for hash-based groupby." << std::endl;
      gdf_error_code = GDF_UNSUPPORTED_METHOD;
  }

  //The following code handles propogating an NVCategory into columns which are of type nvcategory
  if(gdf_error_code == GDF_SUCCESS){
	  for(int key_index = 0; key_index < num_key_columns; key_index++){
		  if(out_key_columns[key_index]->dtype == GDF_STRING_CATEGORY){
			  gdf_error_code = create_nvcategory_from_indices(out_key_columns[key_index],
					  	  	  	  	  	  	  	  	  	  	  in_key_columns[key_index]->dtype_info.category);
			  if(gdf_error_code != GDF_SUCCESS){
				  return gdf_error_code;
			  }
		  }
	  }
	  for(int out_column_index = 0; out_column_index < num_aggregation_columns; out_column_index++){
		  if(out_aggregation_columns[out_column_index]->dtype == GDF_STRING_CATEGORY){
			  gdf_error_code = create_nvcategory_from_indices(out_aggregation_columns[out_column_index],
					  	  	  	  	  	  	  	  	  	  	  in_aggregation_columns[out_column_index]->dtype_info.category);
			  if(gdf_error_code != GDF_SUCCESS){
				  return gdf_error_code;
			  }
		  }
	  }
  }

  POP_RANGE();

  return gdf_error_code;
}
