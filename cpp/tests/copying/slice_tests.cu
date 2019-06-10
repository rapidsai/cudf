/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cudf/copying.hpp>
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/copying/copying_test_helper.hpp>
#include <tests/utilities/nvcategory_utils.cuh>
#include <bitmask/bit_mask.cuh>

void call_slice(gdf_column const*          input_column,
                gdf_index_type const*      indices,
                gdf_size_type              num_indices,
                std::vector<gdf_column*> & output){

  output = cudf::slice(*input_column, indices, num_indices); 
}

struct SliceInputTest : GdfTest {};

TEST_F(SliceInputTest, IndexesNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Perform test
  std::vector<gdf_column*> output;
  ASSERT_NO_THROW(call_slice(input_column.get(), nullptr, 0, output));
  ASSERT_EQ(output.size(), std::size_t(0));
}

TEST_F(SliceInputTest, InputColumnSizeNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  gdf_column input_column{};
  input_column.size = 0;

  // Create indices
  std::vector<gdf_index_type> indices_host{SIZE / 4, SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indices(indices_host);

  // Perform test
  std::vector<gdf_column*> output;
  ASSERT_NO_THROW(call_slice(&input_column, static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output));
  ASSERT_EQ(output.size(), std::size_t(0));
}

TEST_F(SliceInputTest, InputColumnDataNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);
  gdf_column* input_column_test = input_column.get();
  input_column_test->data = nullptr;

  // Create indices
  std::vector<gdf_index_type> indices_host{SIZE / 4, SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indices(indices_host);

  // Perform test
  std::vector<gdf_column*> output;
  ASSERT_ANY_THROW(call_slice(input_column_test, static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output));
}

TEST_F(SliceInputTest, InputColumnBitmaskNull) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);
  gdf_column* input_column_test = input_column.get();
  input_column_test->valid = nullptr;

  // Create indices
  std::vector<gdf_index_type> indices_host{SIZE / 4, SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indices(indices_host);

  // Perform test
  std::vector<gdf_column*> output;
  ASSERT_NO_THROW(call_slice(input_column_test, static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output));
}

TEST_F(SliceInputTest, IndexesSizeNotEven) {
  const int SIZE = 32;
  using ColumnType = std::int32_t;

  // Create input column
  auto input_column = create_random_column<ColumnType>(SIZE);

  // Create indices
  std::vector<gdf_index_type> indices_host{SIZE / 4, SIZE / 2};

  // Create indices for test
  std::vector<gdf_index_type> indices_host_test{SIZE / 4, SIZE / 3, SIZE / 2};
  cudf::test::column_wrapper<gdf_index_type> indices_test(indices_host_test);

  // Perform test
  std::vector<gdf_column*> output;
  ASSERT_ANY_THROW(call_slice(input_column.get(), static_cast<gdf_index_type*>(indices_test.get()->data), indices_test.get()->size, output));
}


template <typename ColumnType>
struct SliceTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(SliceTest, test_types);

/**
 * It performs a parameterized type test, where the array of indices contains
 * multiple values.
 *
 * It tests:
 * when the indices are the same.
 * when is less than 16, less than 64 or greater than 64.
 */
TYPED_TEST(SliceTest, MultipleSlices) {
  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);

  // Create indices
  std::vector<gdf_index_type> indices_host{7, 13, 17, 37, 43, 43, 17, INPUT_SIZE};
  cudf::test::column_wrapper<gdf_index_type> indices(indices_host);

  // Perform operation
  std::vector<gdf_column*> output_column_ptrs;
  ASSERT_NO_THROW(call_slice(input_column.get(), static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output_column_ptrs));

  // Transfer input column to host
  std::vector<TypeParam> input_col_data;
  std::vector<gdf_valid_type> input_col_bitmask;
  std::tie(input_col_data, input_col_bitmask) = input_column.to_host();
  
  // Perform slice in cpu
  std::vector<std::vector<TypeParam>> output_cols_data;
  std::vector<std::vector<gdf_valid_type>> output_cols_bitmask;
  std::vector<gdf_size_type> output_cols_null_count;
  std::tie(output_cols_data, output_cols_bitmask, output_cols_null_count) = slice_columns<TypeParam>(input_col_data, 
                                                                              input_col_bitmask, indices_host);

  // Create Validation output column_wrappers
  std::vector<cudf::test::column_wrapper<TypeParam>> validation_columns;
  for (std::size_t i = 0; i < output_column_ptrs.size(); i++){
    validation_columns.emplace_back(cudf::test::column_wrapper<TypeParam>(output_cols_data[i], output_cols_bitmask[i]));    
    ASSERT_EQ(validation_columns[i].null_count(), output_cols_null_count[i]);
  }

  // Verify the operation
  for (std::size_t i = 0; i < validation_columns.size(); ++i) {
    if (validation_columns[i].size() > 0 && output_column_ptrs[i]->size > 0)
      ASSERT_TRUE(validation_columns[i] == *(output_column_ptrs[i]));
  }

  for (std::size_t i = 0; i < output_column_ptrs.size(); i++){
    gdf_column_free(output_column_ptrs[i]);
    delete output_column_ptrs[i];
  }
}


/**
 * It performs a parameterized type and a parameterized value test. The
 * indices array contains only two values with a fixed length between them.
 * The interval iterates over all the values in the input column.
 */
TYPED_TEST(SliceTest, RangeIndexPosition) {
  // Test parameters
  constexpr gdf_index_type INIT_INDEX{0};
  constexpr gdf_index_type SLICE_RANGE{37};
  constexpr gdf_index_type FINAL_INDEX{INPUT_SIZE - SLICE_RANGE};

  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);
  for (gdf_index_type index = INIT_INDEX; index < FINAL_INDEX; ++index) {
    // Create indices
    std::vector<gdf_index_type> indices_host{index, index + SLICE_RANGE};
    cudf::test::column_wrapper<gdf_index_type> indices(indices_host);
    
    // Perform operation
    std::vector<gdf_column*> output_column_ptrs;
    ASSERT_NO_THROW(call_slice(input_column.get(), static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output_column_ptrs));

    // Transfer input column to host
    std::vector<TypeParam> input_col_data;
    std::vector<gdf_valid_type> input_col_bitmask;
    std::tie(input_col_data, input_col_bitmask) = input_column.to_host();

    // Perform slice in cpu
    std::vector<std::vector<TypeParam>> output_cols_data;
    std::vector<std::vector<gdf_valid_type>> output_cols_bitmask;
    std::vector<gdf_size_type> output_cols_null_count;
    std::tie(output_cols_data, output_cols_bitmask, output_cols_null_count) = slice_columns<TypeParam>(input_col_data, 
                                                                                input_col_bitmask, indices_host);
    
    // Create Validation output column_wrappers
    std::vector<cudf::test::column_wrapper<TypeParam>> validation_columns;
    for (std::size_t i = 0; i < output_column_ptrs.size(); i++){
      validation_columns.emplace_back(cudf::test::column_wrapper<TypeParam>(output_cols_data[i], output_cols_bitmask[i]));    
      ASSERT_EQ(validation_columns[i].null_count(), output_cols_null_count[i]);
    }

    // Verify the operation
    for (std::size_t i = 0; i < validation_columns.size(); ++i) {
      if (validation_columns[i].size() > 0 && output_column_ptrs[i]->size > 0)
        ASSERT_TRUE(validation_columns[i] == *(output_column_ptrs[i]));
    }

    for (std::size_t i = 0; i < output_column_ptrs.size(); i++){
      gdf_column_free(output_column_ptrs[i]);
      delete output_column_ptrs[i];
    }
  }
}

TEST_F(SliceInputTest, NVCategoryMultipleSlices)  {
  
  // Create host strings
  bool print = false;
  const int length = 7;
  const char ** orig_string_data = cudf::test::generate_string_data(INPUT_SIZE, length, print);
  std::vector<std::string> orig_strings_vector(orig_string_data, orig_string_data + INPUT_SIZE);
  
  // Create input column
  gdf_column * input_column = cudf::test::create_nv_category_column_strings(orig_string_data, INPUT_SIZE);

  // Create indices
  std::vector<gdf_index_type> indices_host{7, 13, 17, 37, 43, 43, 17, INPUT_SIZE};
  cudf::test::column_wrapper<gdf_index_type> indices(indices_host);

  // Perform operation
  std::vector<gdf_column*> output_column_ptrs;
  ASSERT_NO_THROW(call_slice(input_column, static_cast<gdf_index_type*>(indices.get()->data), indices.get()->size, output_column_ptrs));

  // Transfer input column to host
  std::vector<std::string> input_col_data;
  std::vector<gdf_valid_type> input_col_bitmask;
  std::tie(input_col_data, input_col_bitmask) = cudf::test::nvcategory_column_to_host(input_column);
  for(gdf_size_type i=0;i<INPUT_SIZE;i++){
    ASSERT_EQ(orig_strings_vector[i], input_col_data[i]);
  }
  
  // Transfer output to host
  std::vector<std::vector<std::string>> host_output_string_vector(output_column_ptrs.size());
  std::vector<std::vector<gdf_valid_type>> host_output_bitmask(output_column_ptrs.size());
  for(std::size_t i=0;i<output_column_ptrs.size();i++){
    std::tie(host_output_string_vector[i], host_output_bitmask[i]) = cudf::test::nvcategory_column_to_host(output_column_ptrs[i]);
  }
  
  // Perform slice in cpu
  std::vector<std::vector<std::string>> output_cols_data;
  std::vector<std::vector<gdf_valid_type>> output_cols_bitmask;
  std::vector<gdf_size_type> output_cols_null_count;
  std::tie(output_cols_data, output_cols_bitmask, output_cols_null_count) = slice_columns<std::string>(input_col_data, 
                                                                              input_col_bitmask, indices_host);

  // Verify the operation
  ASSERT_EQ(host_output_string_vector.size(), output_cols_data.size());
  ASSERT_EQ(host_output_bitmask.size(), output_cols_bitmask.size());
  for (std::size_t i = 0; i < host_output_string_vector.size(); ++i) {
    ASSERT_EQ(host_output_string_vector[i].size(), output_cols_data[i].size());
    ASSERT_EQ(host_output_bitmask[i].size(), output_cols_bitmask[i].size());
    ASSERT_EQ(output_cols_null_count[i], output_column_ptrs[i]->null_count);
    for (std::size_t j = 0; j < host_output_string_vector[i].size(); ++j) {
      ASSERT_EQ(host_output_string_vector[i][j], output_cols_data[i][j]);
    }
    for (std::size_t j = 0; j < host_output_string_vector[i].size(); ++j) {
      bool lhs_is_valids = bit_mask::is_valid(reinterpret_cast<bit_mask::bit_mask_t*>(host_output_bitmask[i].data()),j);
      bool rhs_is_valids = bit_mask::is_valid(reinterpret_cast<bit_mask::bit_mask_t*>(output_cols_bitmask[i].data()),j);
      ASSERT_EQ(lhs_is_valids, rhs_is_valids);
    }
  }

  for (std::size_t i = 0; i < output_column_ptrs.size(); i++){
    gdf_column_free(output_column_ptrs[i]);
    delete output_column_ptrs[i];    
  }
}
