#ifndef REPLACE_DETAIL_HPP
#define REPLACE_DETAIL_HPP

#include <cudf/cudf.h>
#include <cudf/types.hpp>

namespace cudf {
namespace detail {

/**
  * @brief Replaces all null values in a column with corresponding values of another column
  *
  * The first column is expected to be a regular gdf_column. The second column
  * must be of the same type and same size as the first.
  *
  * The function replaces all nulls of the first column with the
  * corresponding elements of the second column
  *
  * @param[in] input A gdf_column containing null values
  * @param[in] replacement A gdf_column whose values will replace null values in input
  * @param[in] stream Optional stream in which to perform allocations
  *
  * @returns gdf_column Column with nulls replaced
  */
gdf_column replace_nulls(const gdf_column& input,
                          const gdf_column& replacement,
                          cudaStream_t stream);

/**
* @brief Replaces all null values in a column with a scalar.
*
* The column is expected to be a regular gdf_column. The scalar is expected to be
* a gdf_scalar of the same data type.
*
* The function will replace all nulls of the column with the scalar value.
*
* @param[in] input A gdf_column containing null values
* @param[in] replacement A gdf_scalar whose value will replace null values in input
* @param[in] inplace Indicates whether to replace nulls inplace
* @param[in] stream Optional stream in which to perform allocations
*
* @returns gdf_column Column with nulls replaced
*/
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement,
                         bool inplace = false,
                         cudaStream_t stream = 0);
}
}

#endif
