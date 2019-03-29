#include "gtest/gtest.h"
#include "copying.hpp"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/copying/copying_test_helper.hpp"

template <typename ColumnType>
struct SliceTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;
TYPED_TEST_CASE(SliceTest, test_types);

/**
 *
 */
TYPED_TEST(SliceTest, MultipleSlices) {
  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);

  // Create indexes
  std::vector<gdf_index_type> indexes_host{7, 13, 17, 37, 17, INPUT_SIZE};
  cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);

  // Create output
  std::vector<std::shared_ptr<cudf::test::column_wrapper<TypeParam>>> output_columns;
  auto source_columns =
      allocate_slice_output_columns<TypeParam>(output_columns, indexes_host);
  cudf::column_array column_array(source_columns.data(), source_columns.size());

  // Perform operation
  ASSERT_NO_THROW(cudf::slice(input_column.get(), indexes.get(), &column_array));

  // Transfer input column to host
  auto input_column_host = makeHelperColumn<TypeParam>(input_column);

  // Transfer output columns to host
  auto output_column_host = makeHelperColumn<TypeParam>(output_columns);

  // Perform split in cpu
  auto output_column_cpu = slice_columns<TypeParam>(input_column_host,
                                                    indexes_host);

  // Verify the operation
  for (std::size_t i = 0; i < output_column_host.size(); ++i) {
    verify<TypeParam>(output_column_cpu[i], output_column_host[i]);
  }
}

/**
 *
 */
TYPED_TEST(SliceTest, RangeIndexPosition) {
  // Test parameters
  constexpr gdf_index_type INIT_INDEX{0};
  constexpr gdf_index_type SLICE_RANGE{37};
  constexpr gdf_index_type FINAL_INDEX{INPUT_SIZE - SLICE_RANGE};

  // Create input column
  auto input_column = create_random_column<TypeParam>(INPUT_SIZE);
  for (gdf_index_type index = INIT_INDEX; index < FINAL_INDEX; ++index) {
    // Create indexes
    std::vector<gdf_index_type> indexes_host{index, index + SLICE_RANGE};
    cudf::test::column_wrapper<gdf_index_type> indexes(indexes_host);
    
    // Create output
    std::vector<std::shared_ptr<cudf::test::column_wrapper<TypeParam>>> output_columns;
    auto source_columns =
        allocate_slice_output_columns<TypeParam>(output_columns, indexes_host);
    cudf::column_array column_array(source_columns.data(), source_columns.size());
    
    // Perform operation
    ASSERT_NO_THROW(cudf::slice(input_column.get(), indexes.get(), &column_array));

    // Transfer input column to host
    auto input_column_host = makeHelperColumn<TypeParam>(input_column);

    // Transfer output columns to host
    auto output_column_host = makeHelperColumn<TypeParam>(output_columns);

    // Perform split in cpu
    auto output_column_cpu = slice_columns<TypeParam>(input_column_host,
                                                      indexes_host);

    // Verify columns
    for (std::size_t i = 0; i < output_column_host.size(); ++i) {
      verify<TypeParam>(output_column_cpu[i], output_column_host[i]);
    }
  }
}
