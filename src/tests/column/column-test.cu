#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/cffi/functions.h>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x01 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x80 ? '1' : '0') 

struct anon {
    __device__ void operator()(int x) { printf("%d ", x); }
};

TEST(gdf_column_concat_test, test1) 
{
    int num_columns = 3;
    std::vector<int32_t> sizes = { 3, 12, 8};

    std::vector<int32_t> input[num_columns];
    std::vector<gdf_valid_type> valid[num_columns];

    thrust::device_vector<int32_t>* d_input[num_columns];
    thrust::device_vector<gdf_valid_type>* d_valid[num_columns];

    gdf_column **input_columns = new gdf_column*[num_columns];

    int total_size = 0;
    for (int i = 0; i < num_columns; ++i) {
        input[i].resize(sizes[i], 0);
        valid[i].resize(gdf_get_num_chars_bitmask(sizes[i]), 254);
        std::generate(input[i].begin(), input[i].end(), [] () { return rand(); });
        std::for_each(input[i].begin(), input[i].end(), [] (int x) { printf("%d ", x);}); printf("\n");
        std::for_each(valid[i].begin(), valid[i].end(), [] (gdf_valid_type x) { printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(x));}); printf("\n");
        d_input[i] = new thrust::device_vector<int32_t>(input[i]);
        d_valid[i] = new thrust::device_vector<gdf_valid_type>(valid[i]);

        input_columns[i] = new gdf_column;

        input_columns[i]->dtype = GDF_INT32;
        input_columns[i]->size = sizes[i];
        total_size += sizes[i];

	    input_columns[i]->data = thrust::raw_pointer_cast(d_input[i]->data());
	    input_columns[i]->valid = thrust::raw_pointer_cast(d_valid[i]->data());
    }
    
    gdf_column output_column;
    int32_t *d_concat_data;
    gdf_valid_type *d_concat_valid;
    cudaMallocManaged(&d_concat_data, sizeof(int32_t)*total_size);
    cudaMallocManaged(&d_concat_valid, sizeof(gdf_valid_type)*total_size / GDF_VALID_BITSIZE);

    gdf_column_view(&output_column, d_concat_data, d_concat_valid, total_size, GDF_INT32);

    cudaDeviceSynchronize();
    printf("Concatenating %d columns\n", num_columns);        
    EXPECT_EQ(GDF_SUCCESS, gdf_column_concat(input_columns, num_columns, &output_column));

    

    for (int i = 0; i < total_size; i++)
        printf("%d ", d_concat_data[i]);
    printf("\n");
    for (size_t i = 0; i < gdf_get_num_chars_bitmask(total_size); ++i)
        printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(d_concat_valid[i]));
    printf("\n");

    cudaFree(d_concat_data);
    cudaFree(d_concat_valid);

    for (int i = 0; i < num_columns; ++i) {
        delete d_input[i];
        delete d_valid[i];
        delete input_columns[i];
    }
    
    delete [] input_columns;
}
