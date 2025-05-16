/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/replace.hpp>

// This is the main test fixture
struct ReplaceTest : public cudf::test::BaseFixture {};

template <typename T>
void normalize_nans_and_zeros_test_internal(
  cudf::test::fixed_width_column_wrapper<T> const& test_data_in,
  cudf::column_view const& test_data_comp)
{
  // mutable overload
  {
    cudf::column test_data(test_data_in);
    cudf::mutable_column_view mutable_view = test_data;
    cudf::normalize_nans_and_zeros(mutable_view);

    // compare bitwise
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      mutable_view.head(), test_data_comp.head(), mutable_view.size() * sizeof(T));
  }

  // returned column overload
  {
    cudf::column test_data(test_data_in);
    cudf::column_view view = test_data;

    auto out      = cudf::normalize_nans_and_zeros(view);
    auto out_view = out->view();

    // compare bitwise
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(
      out_view.head(), test_data_comp.head(), out_view.size() * sizeof(T));
  }
}

// Test for normalize_nans_and_nulls
TEST_F(ReplaceTest, NormalizeNansAndZerosFloat)
{
  // bad data
  cudf::test::fixed_width_column_wrapper<float> f_test_data{
    32.5f, -0.0f, 111.0f, -NAN, NAN, 1.0f, 0.0f, 54.3f};
  // good data
  cudf::test::fixed_width_column_wrapper<float> f_test_data_comp{
    32.5f, 0.0f, 111.0f, NAN, NAN, 1.0f, 0.0f, 54.3f};
  //
  normalize_nans_and_zeros_test_internal<float>(f_test_data, f_test_data_comp);
}

// Test for normalize_nans_and_nulls
TEST_F(ReplaceTest, NormalizeNansAndZerosDouble)
{
  // bad data
  cudf::test::fixed_width_column_wrapper<double> d_test_data{
    32.5, -0.0, 111.0, double(-NAN), double(NAN), 1.0, 0.0, 54.3};
  // good data
  cudf::test::fixed_width_column_wrapper<double> d_test_data_comp{
    32.5, 0.0, 111.0, double(NAN), double(NAN), 1.0, 0.0, 54.3};
  //
  normalize_nans_and_zeros_test_internal<double>(d_test_data, d_test_data_comp);
}

CUDF_TEST_PROGRAM_MAIN()
