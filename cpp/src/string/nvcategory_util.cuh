#ifndef NVCATEGORY_UTIL_H
#define NVCATEGORY_UTIL_H

#include "cudf.h"
#include <vector>
class NVCategory;

const bool DEVICE_ALLOCATED = true;
const bool HOST_ALLOCATED = false;


typedef unsigned int nv_category_index_type;
/**
 * Take a column whose indices map into this nvcategory and generate a condensed copy
 *
 * This function abstracts the usecase where operations have been done to a column like filtering or grouping to
 * create a new column and we want to have a unique nvcategory assigned to it.
 *
 * @param column the column for which we are generating a new NVCategory
 * @param nv_category the category that contains the data these indices map to
 * @return a gdf_error indicating success or failure type
 */
gdf_error create_nvcategory_from_indices(gdf_column * column, NVCategory * nv_category);

/**
 * Take a vector of columns of type GDF_STRING_CATEGORY and merge their dictionaries together in place
 *
 * This function abstracts the use case where a user needs to make columns comparable with each other. It will modify
 * columns that you pass in giving them a new NVCategory which shares a dictionary across all columns.
 * !!!WARNING!!! this function deletes old NVCategory when its done since we will be losing a reference to it.
 * This function should only be used when you have JUST created these columns and the user has not yet had access to
 * them or the NVCategory they contain.
 * @param categories the columns whose dictionaries are going to be merged together.
 * @return a gdf_error indicating success or failure type
 */
gdf_error merge_category_dictionaries(gdf_column * category_columns[], int num_columns);

/**
 * Take a vector of columns of type GDF_STRING_CATEGORY and merge their dictionaries together making copies
 *
 * This function abstracts the use case where a user needs to make columns comparable with each other. It will modify
 * columns that you pass in giving them a new NVCategory which shares a dictionary across all column. This function is
 * safe for to use on data the user is managing.
 * @param input_categories the columns whose dictionaries are going to be merged together.
 * @param output_categories preallocated columns that will store the newly mapped indicies along with the shared dictionary NVCategory
 * @return a gdf_error indicating success or failure type
 */
gdf_error merge_category_dictionaries(gdf_column * input_columns[],gdf_column * output_columns[], int num_columns);


gdf_error concat_categories(gdf_column * input_columns[],gdf_column * output_column, int num_columns);


#endif
