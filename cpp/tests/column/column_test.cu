#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <map>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <cudf/functions.h>

#include "tests/utilities/cudf_test_utils.cuh"

// uncomment to enable benchmarking gdf_column_concat
//#define ENABLE_CONCAT_BENCHMARK 

template <typename T>
struct print {
  __device__ void operator()(T x) { printf("%x ", x); }
};

struct ColumnConcatTest : public testing::Test
{
  ColumnConcatTest() {}
  ~ColumnConcatTest() {}
  
  template <typename T, typename data_initializer_t, typename null_initializer_t>
  void multicolumn_test(std::vector<gdf_size_type> column_sizes, 
                        data_initializer_t data_init, 
                        null_initializer_t null_init)
  { 
    std::vector< std::vector<T> > the_columns(column_sizes.size());

    for (size_t i = 0; i < column_sizes.size(); ++i)
      initialize_vector(the_columns[i], column_sizes[i], data_init);
    
    // This is just an alias to a gdf_column with a custom deleter that will free 
    // the data and valid fields when the unique_ptr goes out of scope
    using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>; 

    // Copies the random data from each host vector in the_columns to the device in a gdf_column. 
    // Each gdf_column's validity bit i will be initialized with the lambda
    std::vector<gdf_col_pointer> gdf_columns = initialize_gdf_columns(the_columns, null_init);

    std::vector<gdf_column*> raw_gdf_columns;

    for(auto const & c : gdf_columns) {
      raw_gdf_columns.push_back(c.get());
    }

    gdf_column **columns_to_concat = raw_gdf_columns.data();

    int num_columns = raw_gdf_columns.size();
    gdf_size_type total_size = 0;
    for (auto sz : column_sizes) total_size += sz;

    std::vector<T> output_data(total_size);
    std::vector<gdf_valid_type> output_valid(gdf_get_num_chars_bitmask(total_size));
    
    auto output_gdf_col = create_gdf_column(output_data, output_valid);

    EXPECT_EQ( GDF_SUCCESS, gdf_column_concat(output_gdf_col.get(), 
                                              columns_to_concat, 
                                              num_columns) );

    // make a concatenated reference
    std::vector<T> ref_data;
    for (size_t i = 0; i < the_columns.size(); ++i)
      std::copy(the_columns[i].begin(), the_columns[i].end(), std::back_inserter(ref_data));
      
    gdf_size_type ref_null_count = 0;
    std::vector<gdf_valid_type> ref_valid(gdf_get_num_chars_bitmask(total_size));
    for (gdf_size_type index = 0, col = 0, row = 0; index < total_size; ++index)
    {
      if (null_init(row, col)) gdf::util::turn_bit_on(ref_valid.data(), index);
      else ref_null_count++;
      
      if (++row >= column_sizes[col]) { row = 0; col++; }
    }   
    auto ref_gdf_col = create_gdf_column(ref_data, ref_valid);

    EXPECT_EQ(ref_null_count, ref_gdf_col->null_count);

    EXPECT_TRUE(gdf_equal_columns<int>(ref_gdf_col.get(), output_gdf_col.get()));

    //print_valid_data(ref_valid.data(), total_size); printf("\n");
    //print_valid_data(output_gdf_col->valid, total_size);
  }

  template <typename T, typename data_initializer_t, typename null_initializer_t>
  void multicolumn_bench(std::vector<size_t> column_sizes, 
                         data_initializer_t data_init, 
                         null_initializer_t null_init)
  {
    std::vector< std::vector<T> > the_columns(column_sizes.size());

    for (size_t i = 0; i < column_sizes.size(); ++i)
      initialize_vector(the_columns[i], column_sizes[i], data_init);

    std::vector<gdf_col_pointer> gdf_columns = initialize_gdf_columns(the_columns, null_init);

    std::vector<gdf_column*> raw_gdf_columns;

    for(auto const & c : gdf_columns) {
      raw_gdf_columns.push_back(c.get());
    }

    gdf_column **columns_to_concat = raw_gdf_columns.data();

    int num_columns = raw_gdf_columns.size();
    gdf_size_type total_size = 0;
    for (auto sz : column_sizes) total_size += sz;

    std::vector<int32_t> output_data(total_size);
    std::vector<gdf_valid_type> output_valid(gdf_get_num_chars_bitmask(total_size));
    
    auto output_gdf_col = create_gdf_column(output_data, output_valid);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EXPECT_EQ( GDF_SUCCESS, gdf_column_concat(output_gdf_col.get(), 
                                              columns_to_concat, 
                                              num_columns) );

    int num = 100;
    for (int i = 0; i < num; ++i) {
      gdf_column_concat(output_gdf_col.get(), columns_to_concat, num_columns);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time for " << num << " concats of " << num_columns << " columns of " 
              << total_size  << " total elements:\n";
    std::cout << diff.count() << " s\n";
  }
};

// Test various cases with null pointers or empty columns
TEST_F(ColumnConcatTest, ErrorConditions)
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

TEST_F(ColumnConcatTest, RandomData) {
  gdf_size_type column_size = 1005;
  gdf_size_type null_interval = 17;
    
  std::vector<gdf_size_type> column_sizes{column_size, column_size, column_size};

  multicolumn_test<int>(column_sizes, 
                        [](int index){ return std::rand(); },
                        [null_interval](gdf_size_type row, gdf_size_type col) { 
                          return (row % null_interval) != 0; 
                        });
}
  
TEST_F(ColumnConcatTest, DifferentLengthColumns) {
  gdf_size_type null_interval = 2;
    
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  multicolumn_test<int>(column_sizes, 
                        [](int index){ return std::rand(); },
                        [null_interval](gdf_size_type row, gdf_size_type col) { 
                          return (row % null_interval) != 0; 
                        });
}

TEST_F(ColumnConcatTest, DifferentLengthColumnsLimitedBits) {   
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  auto limited_bits = [column_sizes](gdf_size_type row, gdf_size_type col){ 
    return row < column_sizes[col]; 
  };

  multicolumn_test<int>(column_sizes, 
                        [](int index){ return std::rand(); },
                        limited_bits);
}

TEST_F(ColumnConcatTest, MoreComplicatedColumns) {   
   
  std::vector<gdf_size_type> column_sizes{5, 1003, 17, 117};

  auto bit_setter = [column_sizes](gdf_size_type row, gdf_size_type col) { 
    switch (col) {
    case 0: 
      return (row % 2) != 0; // column 0 has odd bits set
    case 1:
      return row < column_sizes[col];
    case 2:
      return (row % 17) != 0; 
    case 3:
      return row < 3;
    }
    return true;
  };

  multicolumn_test<int>(column_sizes, 
                        [](int index){ return std::rand(); },
                        bit_setter);
}


TEST_F(ColumnConcatTest, EightByteColumns) {   
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  auto limited_bits = [column_sizes](gdf_size_type row, gdf_size_type col){ 
    return row < column_sizes[col]; 
  };

  multicolumn_test<int64_t>(column_sizes, 
                            [](int index){ return std::rand(); },
                            limited_bits);
}


#ifdef ENABLE_CONCAT_BENCHMARK
TEST_F(ColumnConcatTest, Benchmark) {   
   
  size_t n = 42000000;
  std::vector<size_t> column_sizes{n, n, n, n};

  gdf_size_type null_interval = 17;

  auto bit_setter = [null_interval](gdf_size_type row, gdf_size_type col) { 
    return (row % null_interval) != 0; 
  };

  multicolumn_bench<int>(column_sizes, 
                        [](int index){ return std::rand(); },
                        bit_setter);
}
#endif // ENABLE_CONCAT_BENCHMARK


TEST(ColumnByteWidth, TestByteWidth)
{

  std::map<gdf_dtype, int> enum_to_type_size { {GDF_INT8, sizeof(int8_t)},
                                                  {GDF_INT16, sizeof(int16_t)},
                                                  {GDF_INT32, sizeof(int32_t)},
                                                  {GDF_INT64, sizeof(int64_t)},
                                                  {GDF_FLOAT32, sizeof(float)},
                                                  {GDF_FLOAT64, sizeof(double)},
                                                  {GDF_DATE32, sizeof(gdf_date32)},
                                                  {GDF_DATE64, sizeof(gdf_date64)},
                                                  {GDF_TIMESTAMP, sizeof(gdf_timestamp)},
                                                  {GDF_CATEGORY, sizeof(gdf_category)}
                                                };
  for(auto const& pair : enum_to_type_size)
  {
    int byte_width{0};
    gdf_column col;
    col.dtype = pair.first;
    ASSERT_EQ(GDF_SUCCESS, get_column_byte_width(&col, &byte_width));
    EXPECT_EQ(pair.second, byte_width);
  }
}
