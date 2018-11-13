/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include "gdf_table.cuh"
#include <gdf/gdf.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <memory>
#include <stdio.h>

gdf_error gdf_transpose(size_t ncols,
                        gdf_column** in_cols,
                        gdf_column** out_cols)
{
    // Make sure the inputs are not null
    if( (0 == ncols) 
        || (nullptr == in_cols) 
        || (nullptr == out_cols))
    {
        return GDF_DATASET_EMPTY;
    }

    // If there are no rows in the input, return successfully
    if ( 0 == in_cols[0]->size )
    {
        return GDF_SUCCESS;
    }

    // Wrap the input columns in a gdf_table
    using size_type = decltype(ncols);
    size_t out_ncols = in_cols[0]->size;
    std::unique_ptr< const gdf_table<size_type> > input_table {new gdf_table<size_type>(ncols, in_cols)};
    std::unique_ptr< gdf_table<size_type> > output_table {new gdf_table<size_type>(out_ncols, out_cols)};

    // Workaround because device lambdas cannot currently work with smart pointers
    // Smart pointers are still used because I don't want to remember to free memory
    auto input_table_ptr = &(*input_table);
    auto output_table_ptr = &(*output_table);
    auto copy_to_outcol = [=] __device__ (size_t i)
    {
        input_table_ptr->get_packed_row_values(i, 
            (unsigned char*) output_table_ptr->get_column_device_pointer(i));
        input_table_ptr->get_row_valids(i, 
            (unsigned char*) output_table_ptr->get_columns_device_valids_ptr(i));        
    };

    thrust::for_each(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(out_ncols), 
                    copy_to_outcol);
    return GDF_SUCCESS;
}