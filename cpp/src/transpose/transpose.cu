/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include "dataframe/cudf_table.cuh"
#include <cudf.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <memory>
#include <stdio.h>

gdf_error gdf_transpose(gdf_size_type ncols,
                        gdf_column** in_cols,
                        gdf_column** out_cols)
{
    // Make sure the inputs are not null
    GDF_REQUIRE( (ncols > 0) && (nullptr != in_cols) && (nullptr != out_cols), GDF_DATASET_EMPTY)

    // If there are no rows in the input, return successfully
    GDF_REQUIRE( in_cols[0]->size > 0, GDF_SUCCESS)

    // Check datatype homogeneity
    gdf_dtype dtype = in_cols[0]->dtype; 
    for(gdf_size_type i = 1; i < ncols; i++)
    {
        GDF_REQUIRE(in_cols[i]->dtype == dtype, GDF_DTYPE_MISMATCH)
    }

    // Check if there are nulls to be processed
    bool has_null = false;
    for(gdf_size_type i = 0; i < ncols; i++)
    {
        if (in_cols[i]->null_count > 0) {
            has_null = true;
            break;
        }   
    }
    gdf_size_type out_ncols = in_cols[0]->size;
    if (has_null)
        for(gdf_size_type i = 0; i < out_ncols; i++)
            GDF_REQUIRE(out_cols[i]->valid != nullptr, GDF_DATASET_EMPTY)

    // Wrap the input columns in a gdf_table
    using size_type = decltype(ncols);
    std::unique_ptr< const gdf_table<size_type> > input_table {new gdf_table<size_type>(ncols, in_cols)};
    std::unique_ptr< gdf_table<size_type> > output_table {new gdf_table<size_type>(out_ncols, out_cols)};

    // Workaround because device lambdas cannot currently work with smart pointers
    // Smart pointers are still used because I don't want to remember to free memory
    auto input_table_ptr = input_table.get();
    auto output_table_ptr = output_table.get();
    auto copy_to_outcol = [input_table_ptr, output_table_ptr, has_null] __device__ (gdf_size_type i)
    {
        input_table_ptr->get_packed_row_values(i, 
            output_table_ptr->get_column_device_pointer(i));
        
        if (has_null) {
            input_table_ptr->get_row_valids(i, 
                output_table_ptr->get_columns_device_valids_ptr(i));
        } 
    };

    thrust::for_each(thrust::counting_iterator<gdf_size_type>(0),
                    thrust::counting_iterator<gdf_size_type>(out_ncols), 
                    copy_to_outcol);
    return GDF_SUCCESS;
}