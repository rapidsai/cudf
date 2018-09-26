#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/cffi/functions.h>

#include "../test_utils/gdf_test_utils.cuh"

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

template <typename T>
struct anon {
  __device__ void operator()(T x) { printf("%x ", x); }
};

// Test various cases with null pointers or empty columns
TEST(ColumnConcatTest, EmptyData)
{
  constexpr int num_columns = 4;

  // Test null output column
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(0, 0, 0));
  
  std::vector<gdf_size_type> column_sizes{4, 1, 2, 3};
  
  gdf_size_type total_size = 0;
  for (auto& n : column_sizes)
    total_size += n;
  
  gdf_column **input_columns = new gdf_column*[num_columns];
  for (int i = 0; i < num_columns; ++i) {
    input_columns[i] = 0;
  }

  std::vector<int32_t> output_data(total_size);
  std::vector<gdf_valid_type> output_valid(gdf_get_num_chars_bitmask(total_size));
  
  auto output_gdf_col = create_gdf_column(output_data, output_valid);
  
  // Test array of null input columns
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(output_gdf_col.get(), input_columns, num_columns));

  for (int i = 0; i < num_columns; ++i) {
    gdf_column col;
    EXPECT_EQ(GDF_SUCCESS, gdf_column_view(&col, 0, 0, column_sizes[i], GDF_INT32));
    input_columns[i] = &col;
  }

  // test null input column data / valid pointers
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(output_gdf_col.get(), input_columns, num_columns));

  // create some actual input columns
  for (int i = 0; i < num_columns; ++i) {
    gdf_size_type size = column_sizes[i];
    std::vector<int32_t> data(size);
    std::vector<gdf_valid_type> valid(gdf_get_num_chars_bitmask(size));
  
    input_columns[i] = create_gdf_column(data, valid).get();
  }

  // test mismatched sizes
  output_gdf_col->size = total_size - 1;
  EXPECT_EQ(GDF_COLUMN_SIZE_MISMATCH, gdf_column_concat(output_gdf_col.get(), input_columns, num_columns));
}

TEST(ColumnConcatTest, RandomData) {
  // VTuple is a parameter pack for a std::tuple of vectors, 
  // this is the only way I could come up with for having a container of vectors of different types
  using multi_col_t = VTuple<int, int, int>; 

  // the_columns is now the same as a tuple<vector<int>, vector<int>, vector<int> >
  multi_col_t the_columns; 
  gdf_size_type column_size = 1005;
  gdf_size_type null_every = 17;
  
  // Initializes each vector to length 1000 with random data
  initialize_tuple(the_columns, column_size, [](int index){ return std::rand(); }); 

  // This is just an alias to a gdf_column with a custom deleter that will free 
  // the data and valid fields when the unique_ptr goes out of scope
  using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>; 

   // Copies the random data from each host vector in the_columns to the device in a gdf_column. 
   // Each gdf_column's validity bit i will be initialized with the lambda
  std::vector<gdf_col_pointer> gdf_columns = initialize_gdf_columns(the_columns, 
                                                                    [null_every](gdf_size_type index, size_t){ return ((index % null_every) != 0); });

  std::vector<gdf_column*> raw_gdf_columns;

  for(auto const & c : gdf_columns) {
    raw_gdf_columns.push_back(c.get());
  }

  gdf_column **columns_to_concat = raw_gdf_columns.data();

  int num_columns = raw_gdf_columns.size();
  gdf_size_type total_size = column_size * num_columns;

  std::vector<int32_t> output_data(total_size);
  std::vector<gdf_valid_type> output_valid(gdf_get_num_chars_bitmask(total_size));
  
  auto output_gdf_col = create_gdf_column(output_data, output_valid);

  EXPECT_EQ( GDF_SUCCESS, gdf_column_concat(output_gdf_col.get(), 
                                            columns_to_concat, 
                                            num_columns) );

  // make a concatenated reference
  std::vector<int32_t> ref_data;  
  std::copy(std::get<0>(the_columns).begin(), std::get<0>(the_columns).end(), std::back_inserter(ref_data));
  std::copy(std::get<1>(the_columns).begin(), std::get<1>(the_columns).end(), std::back_inserter(ref_data));
  std::copy(std::get<2>(the_columns).begin(), std::get<2>(the_columns).end(), std::back_inserter(ref_data));
    
  std::vector<gdf_valid_type> ref_valid(gdf_get_num_chars_bitmask(total_size));
  for (gdf_size_type index = 0, row = 0; index < total_size; ++index)
  {
    if (row % null_every) gdf::util::turn_bit_on(ref_valid.data(), index);
    if (++row >= column_size) row = 0;
  }   
  auto ref_gdf_col = create_gdf_column(ref_data, ref_valid);

  EXPECT_EQ(num_columns * ((column_size + null_every-1) / null_every), ref_gdf_col->null_count);

  EXPECT_TRUE(gdf_equal_columns<int>(ref_gdf_col.get(), output_gdf_col.get()));

  //std::for_each(ref_valid.begin(), ref_valid.end(), [] (gdf_valid_type x) { printf("%x ", x);}); printf("\n");
  //thrust::for_each(thrust::cuda::par, output_gdf_col->valid, output_gdf_col->valid + ref_valid.size(), anon<gdf_valid_type>()); printf("\n");
  // cudaDeviceSynchronize();

}
  
