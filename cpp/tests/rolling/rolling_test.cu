/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>

#include <cudf.h>
#include <utilities/error_utils.hpp>
#include <utilities/cudf_utils.h>
#include <utilities/column_wrapper.cuh>

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>

template <typename T>
class RollingTest : public GdfTest {

protected:
  void make_random_input(size_t nrows, bool add_nulls)
  {
    // generate random numbers for the vector
    std::mt19937 rng(1);
    auto generator = [&](){ 
      return rng() % std::numeric_limits<T>::max() + 1;
    };

    in_col.resize(nrows);
    std::generate(in_col.begin(), in_col.end(), generator);

    // generate a random validity mask
    if (add_nulls) {
      auto valid_generator = [&](){ 
        return static_cast<bool>((rng() % std::numeric_limits<T>::max()) % 2);
      };
     
      in_col_valid.resize(nrows);
      std::generate(in_col_valid.begin(), in_col_valid.end(), valid_generator);
    }
  }

  void create_gdf_input_buffers()
  {
    if (in_col_valid.size() > 0) {
      auto valid_generator = [&](size_t row, size_t col){
        return in_col_valid[row];
      };
      in_gdf_col = init_gdf_column(in_col, 0, valid_generator);
    }
    else
      in_gdf_col = create_gdf_column(in_col);
  }

  void create_gdf_output_buffers()
  {
    // TODO: do we always need to create a valid buffer? we can't guarantee that every entry will be valid up front 
    auto valid_generator = [&](size_t row, size_t col){ return true; };
    std::vector<T> out_col(in_col.size());
    out_gdf_col = init_gdf_column(out_col, 0, valid_generator);
  }

  void create_reference_output(size_t window, size_t min_periods, size_t forward_window, gdf_agg_op agg)
  {
    // compute the reference solution on the cpu
    size_t nrows = in_col.size();
    ref_data.resize(nrows);
    ref_data_valid.resize(nrows);
    bool has_nulls = true;	// always have the valid bit mask set
    for(size_t i = 0; i < nrows; i++) {
      ASSERT_TRUE(agg == GDF_SUM);
      // TODO: need a more generic way to handle aggregators - initialize (check groupby test)
      T val = 0;
      size_t count = 0;
      // compute bounds
      size_t start_index = std::max((size_t)0, i - window + 1);
      size_t end_index = std::min(nrows, i + forward_window + 1);	// exclusive
      // aggregate
      for (size_t j = start_index; j < end_index; j++) {
        if (in_col_valid[j]) {
          // TODO: need a more generic way to handle aggregators (maybe type dispatcher?)
          val = val + in_col[j];
          count++;
        }
      }
      ref_data[i] = val;
      ref_data_valid[i] = (count >= min_periods);
    }
  }

  void compare_gdf_result()
  {
    // convert to column_wrapper due to a bug in gdf_equal_columns
    // see https://github.com/rapidsai/cudf/issues/1248 for more detail
    // TODO: uncomment when the bug is fixed
    #ifdef BUG_1248_FIXED
      auto valid_generator = [&](size_t row, size_t col){
        return ref_data_valid[row];
      };
      gdf_col_pointer ref_gdf_col = init_gdf_column(ref_data, 0, valid_generator);
      ASSERT_TRUE(gdf_equal_columns<T>(ref_gdf_col.get(), out_gdf_col.get()));
    #else
      // copy output data to host
      gdf_size_type nrows = in_col.size();
      std::vector<T> out_col(nrows);
      cudaMemcpy(out_col.data(), static_cast<T*>(out_gdf_col->data), nrows * sizeof(T), cudaMemcpyDefault);
      
      // copy output valid mask to host
      gdf_size_type nmasks = gdf_valid_allocation_size(nrows);
      std::vector<gdf_valid_type> out_col_mask(nmasks);
      cudaMemcpy(out_col_mask.data(), static_cast<gdf_valid_type*>(out_gdf_col->valid), nmasks * sizeof(gdf_valid_type), cudaMemcpyDefault);
      
      // create column wrappers and compare
      cudf::test::column_wrapper<T> out(out_col, [&](gdf_index_type i) { return gdf_is_valid(out_col_mask.data(), i); } );
      cudf::test::column_wrapper<T> ref(ref_data, [&](gdf_index_type i) { return ref_data_valid[i]; } );
      ASSERT_TRUE(out == ref);
    #endif
  }

  void run_test(size_t window, size_t min_periods, size_t forward_window, gdf_agg_op agg)
  {
    create_gdf_input_buffers();
    create_gdf_output_buffers();

    gdf_rolling_window(out_gdf_col.get(), in_gdf_col.get(),
		       window, min_periods, forward_window, agg,
		       NULL, NULL, NULL);

    create_reference_output(window, min_periods, forward_window, agg);

    compare_gdf_result();
  }

  // input
  std::vector<T> in_col;
  std::vector<bool> in_col_valid;
  gdf_col_pointer in_gdf_col;

  // reference
  std::vector<T> ref_data;
  std::vector<bool> ref_data_valid;

  // output
  gdf_col_pointer out_gdf_col;
};

using TestTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;

TYPED_TEST_CASE(RollingTest, TestTypes);

TYPED_TEST(RollingTest, Simple)
{
  // simple example from Pandas docs:
  //   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  this->in_col 	     = {0, 1, 2, 0, 4};
  this->in_col_valid = {1, 1, 1, 0, 1};

  this->run_test(2, 2, 0, GDF_SUM);
}
