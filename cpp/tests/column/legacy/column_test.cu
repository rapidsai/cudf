#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>

#include <utilities/cudf_utils.h>
#include <utilities/column_utils.hpp>
#include <cudf/cudf.h>

#include <thrust/device_vector.h>



#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <map>


// uncomment to enable benchmarking gdf_column_concat
//#define ENABLE_CONCAT_BENCHMARK 

template <typename T>
struct print {
  __device__ void operator()(T x) { printf("%x ", x); }
};

template <typename ColumnType>
struct ColumnConcatTest : public cudf::test::BaseFixture
{
  ColumnConcatTest() {}
  ~ColumnConcatTest() {}
  
  template <typename data_initializer_t, typename null_initializer_t>
  void multicolumn_test(std::vector<gdf_size_type> column_sizes, 
                        data_initializer_t data_init, 
                        null_initializer_t null_init)
  { 
    std::vector< std::vector<ColumnType> > the_columns(column_sizes.size());

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

    std::vector<ColumnType> output_data(total_size);
    std::vector<gdf_valid_type> output_valid(gdf_valid_allocation_size(total_size));
    
    auto output_gdf_col = create_gdf_column(output_data, output_valid);

    EXPECT_EQ( GDF_SUCCESS, gdf_column_concat(output_gdf_col.get(), 
                                              columns_to_concat, 
                                              num_columns) );

    // make a concatenated reference
    std::vector<ColumnType> ref_data;
    for (size_t i = 0; i < the_columns.size(); ++i)
      std::copy(the_columns[i].begin(), the_columns[i].end(), std::back_inserter(ref_data));
      
    gdf_size_type ref_null_count = 0;
    std::vector<gdf_valid_type> ref_valid(gdf_valid_allocation_size(total_size));
    for (gdf_size_type index = 0, col = 0, row = 0; index < total_size; ++index)
    {
      if (null_init(row, col)) cudf::util::turn_bit_on(ref_valid.data(), index);
      else ref_null_count++;
      
      if (++row >= column_sizes[col]) { row = 0; col++; }
    }   
    auto ref_gdf_col = create_gdf_column(ref_data, ref_valid);

    EXPECT_EQ(ref_null_count, ref_gdf_col->null_count);

    EXPECT_TRUE(gdf_equal_columns(*ref_gdf_col.get(), *output_gdf_col.get()));

  }

  template <typename data_initializer_t, typename null_initializer_t>
  void multicolumn_bench(std::vector<size_t> column_sizes, 
                         data_initializer_t data_init, 
                         null_initializer_t null_init)
  {
    std::vector< std::vector<ColumnType> > the_columns(column_sizes.size());

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

    std::vector<ColumnType> output_data(total_size);
    std::vector<gdf_valid_type> output_valid(gdf_valid_allocation_size(total_size));
    
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

using TestTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(ColumnConcatTest, TestTypes);

TYPED_TEST(ColumnConcatTest, ZeroColumns){
  EXPECT_EQ(GDF_INVALID_API_CALL, gdf_column_concat(nullptr, nullptr, 0));
}

TYPED_TEST(ColumnConcatTest, NegativeColumns){
  EXPECT_EQ(GDF_INVALID_API_CALL, gdf_column_concat(nullptr, nullptr, -1));
}

TYPED_TEST(ColumnConcatTest, NullOutput){
  gdf_column input{};
  gdf_column * input_p = &input;
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(nullptr, &input_p, 1));
}

TYPED_TEST(ColumnConcatTest, NullInput){
  gdf_column output{};
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(&output, nullptr, 1));
}

TYPED_TEST(ColumnConcatTest, NullFirstInputColumn){
  gdf_column output{};
  gdf_column * input_p = nullptr;
  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(&output, &input_p, 1));
}

TYPED_TEST(ColumnConcatTest, OutputWrongSize){
  gdf_size_type num_columns = 4;
  std::vector<gdf_size_type> column_sizes{4, 1, 2, 3};
  ASSERT_EQ(num_columns, static_cast<gdf_size_type>(column_sizes.size()));

  gdf_size_type const total_size{
      std::accumulate(column_sizes.begin(), column_sizes.end(), 0)};

  std::vector<gdf_col_pointer> input_column_pointers(num_columns);
  std::vector<gdf_column*> input_columns(num_columns, nullptr);

  for (int i = 0; i < num_columns; ++i) {
    gdf_size_type size = column_sizes[i];
    std::vector<TypeParam> data(size);
    std::vector<gdf_valid_type> valid(gdf_valid_allocation_size(size));
    input_column_pointers[i] = create_gdf_column(data, valid);
    input_columns[i] = input_column_pointers[i].get();
  }
  std::vector<TypeParam> output_data(total_size);
  std::vector<gdf_valid_type> output_valid(gdf_valid_allocation_size(total_size));
  auto output_gdf_col = create_gdf_column(output_data, output_valid);

  // test mismatched sizes
  output_gdf_col->size = total_size - 1;
  EXPECT_EQ(GDF_COLUMN_SIZE_MISMATCH, gdf_column_concat(output_gdf_col.get(), input_columns.data(), num_columns));
}

TYPED_TEST(ColumnConcatTest, NullInputData){
  gdf_size_type num_columns = 4;
  std::vector<gdf_size_type> column_sizes{4, 1, 2, 3};
  ASSERT_EQ(num_columns, static_cast<gdf_size_type>(column_sizes.size()));

  gdf_size_type const total_size{
      std::accumulate(column_sizes.begin(), column_sizes.end(), 0)};

  std::vector<TypeParam> output_data(total_size);
  std::vector<gdf_valid_type> output_valid(gdf_valid_allocation_size(total_size));
  auto output_gdf_col = create_gdf_column(output_data, output_valid);

  std::vector<gdf_column> cols(num_columns);
  std::vector<gdf_column*> input_columns(num_columns, nullptr);
  for (int i = 0; i < num_columns; ++i) {
    cols[i].data = nullptr;
    cols[i].valid = nullptr;
    cols[i].size = column_sizes[i];
    cols[i].dtype = output_gdf_col->dtype;
    input_columns[i] = &cols[i];
  }

  EXPECT_EQ(GDF_DATASET_EMPTY, gdf_column_concat(output_gdf_col.get(), input_columns.data(), num_columns));
}

TYPED_TEST(ColumnConcatTest, RandomData) {
  gdf_size_type column_size = 1005;
  gdf_size_type null_interval = 17;
    
  std::vector<gdf_size_type> column_sizes{column_size, column_size, column_size};

  this->multicolumn_test(column_sizes, 
                        [](int index){ return std::rand(); },
                        [null_interval](gdf_size_type row, gdf_size_type col) { 
                          return (row % null_interval) != 0; 
                        });
}
  
TYPED_TEST(ColumnConcatTest, DifferentLengthColumns) {
  gdf_size_type null_interval = 2;
    
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  this->multicolumn_test(column_sizes, 
                        [](int index){ return std::rand(); },
                        [null_interval](gdf_size_type row, gdf_size_type col) { 
                          return (row % null_interval) != 0; 
                        });
}

TYPED_TEST(ColumnConcatTest, DifferentLengthColumnsLimitedBits) {   
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  auto limited_bits = [column_sizes](gdf_size_type row, gdf_size_type col){ 
    return row < column_sizes[col]; 
  };

  this->multicolumn_test(
      column_sizes, [](int index) { return std::rand(); }, limited_bits);
}

TYPED_TEST(ColumnConcatTest, MoreComplicatedColumns) {   
   
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

  this->multicolumn_test(column_sizes, 
                        [](int index){ return std::rand(); },
                        bit_setter);
}


TYPED_TEST(ColumnConcatTest, EightByteColumns) {   
  std::vector<gdf_size_type> column_sizes{13, 3, 5};

  auto limited_bits = [column_sizes](gdf_size_type row, gdf_size_type col){ 
    return row < column_sizes[col]; 
  };

  this->multicolumn_test(column_sizes, 
                            [](int index){ return std::rand(); },
                            limited_bits);
}


TYPED_TEST(ColumnConcatTest, SingleColumn){
  std::vector<gdf_size_type> column_sizes{13};
  this->multicolumn_test(column_sizes, 
                        [](int index){ return std::rand(); },
                        [](gdf_size_type row, gdf_size_type col) { 
                          return true; 
                        });
}


#ifdef ENABLE_CONCAT_BENCHMARK
TYPED_TEST(ColumnConcatTest, Benchmark) {   
   
  size_t n = 42000000;
  std::vector<size_t> column_sizes{n, n, n, n};

  gdf_size_type null_interval = 17;

  auto bit_setter = [null_interval](gdf_size_type row, gdf_size_type col) { 
    return (row % null_interval) != 0; 
  };

  multicolumn_bench<TypeParam>(column_sizes, 
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
    gdf_column col{};
    col.dtype = pair.first;
    ASSERT_NO_THROW(byte_width = cudf::byte_width(col));
    EXPECT_EQ(pair.second, byte_width);
  }
}

TEST(ColumnByteWidth, TestGdfTypeSize)
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
    EXPECT_EQ(pair.second, (int) cudf::size_of(pair.first));
  }
}
