#ifndef NVCATEGORY_UTIL_H
#define NVCATEGORY_UTIL_H

#include "cudf.h"
#include <vector>
class NVCategory;

const bool DEVICE_ALLOCATED = true;
const bool HOST_ALLOCATED = false;


typedef int nv_category_index_type;
/**
 * @brief Create a condensed copy of an nvcategory using a column of indices
 *
 * This function abstracts the usecase of gathering indices from an NVCategory.
 *
 * @param[in,out] column the column for which we are generating a new NVCategory
 * @param[in] nv_category the category that contains the data these indices map to
 * @return a gdf_error indicating success or failure type
 */
gdf_error nvcategory_gather(gdf_column * column, NVCategory * nv_category);

/**
 * @brief Takes an array of input_columns and concatenates them into one long column.
 *
 * @param[in] input_columns The Columns to concat.
 * @param[out] Concatted output
 * @param[in] number of input columns
 */
gdf_error concat_categories(gdf_column * input_columns[],gdf_column * output_column, int num_columns);

/**
 * @brief Takes an array of input_columns and makes it so that they all share the same keys in NVCategory
 *
 * @param[in] the columns whose categories must be synchronized
 * @param[out] same data as input_columns but with categories syncrhonized
 * @param[in] number of input columns
 */
gdf_error sync_column_categories(gdf_column * input_columns[],gdf_column * output_columns[], int num_columns);

#endif
