#include "new_groupby.hpp"


namespace{
  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Verifies that a set of columns contain non-null buffers, and are all 
   * of the same size.
   * 
   * @Param first 
   * @Param last
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  gdf_error verify_columns(gdf_column * first, gdf_column * last)
  {
    GDF_REQUIRE((nullptr != first), GDF_DATASET_EMPTY);

    int const required_size{first->size};

    for(; first != last; ++first)
    {
      GDF_REQUIRE(nullptr != first, GDF_DATASET_EMPTY);

      GDF_REQUIRE(required_size == first->size, GDF_COLUMN_SIZE_MISMATCH );

      // TODO Remove when null support for hash-based groupby is added
      GDF_REQUIRE(nullptr == first->valid || 0 == first->null_count, GDF_VALIDITY_UNSUPPORTED);
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
      || (nullptr == options)
      || (nullptr == out_key_columns)
      || (nullptr == out_aggregation_columns)
    )
  {
    return GDF_DATASET_EMPTY;
  }

  // Return immediately if inputs are empty
  GDF_REQUIRE(0 != in_key_columns[0]->size, GDF_SUCCESS);
  GDF_REQUIRE(0 != in_aggregation_columns[0]->size, GDF_SUCCESS);

  auto result = verify_columns(in_key_columns, in_key_columns + num_key_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  result = verify_columns(in_aggregation_columns, in_aggregation_columns + num_aggregation_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_GROUPBY", GROUPBY_COLOR);


}
