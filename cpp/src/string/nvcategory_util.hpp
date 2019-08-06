#ifndef NVCATEGORY_UTIL_H
#define NVCATEGORY_UTIL_H

#include <cudf/cudf.h>
#include <vector>

// Forward declarations
class NVCategory;
namespace cudf{
    struct table;
}

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
gdf_error nvcategory_gather(gdf_column* column, NVCategory* nv_category);

/**
 * @brief Takes an array of input_columns and concatenates them into one long column.
 *
 * @param[in] input_columns The Columns to concat.
 * @param[out] Concatted output
 * @param[in] number of input columns
 */
gdf_error concat_categories(const gdf_column* const input_columns[], gdf_column* output_column, int num_columns);

/**
 * @brief Takes an array of input_columns and makes it so that they all share the same keys in NVCategory
 *
 * @param[in] input_columns the columns whose categories must be synchronized
 * @param[out] output_columns same data as input_columns but with categories syncrhonized
 * @param[in] num_columns number of input columns
 */
gdf_error sync_column_categories(const gdf_column* const input_columns[], gdf_column* output_columns[], int num_columns);

/**
 * @brief Takes two tables and gathers the destination table's data interpreted as int32 from the dictionary of the source table's NVCategory.
 *
 * @param[in] source_table Contains columns that contain dictionaries used for gathering.
 * @param[in,out] destination_table Contains columns that contain indices that map into source_table dictionaries.
 */
gdf_error nvcategory_gather_table(cudf::table source_table, cudf::table destination_table);

#endif
