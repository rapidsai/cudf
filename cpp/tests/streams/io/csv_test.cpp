/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/table/table_view.hpp>

#include <string>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class CSVTest : public cudf::test::BaseFixture {};

TEST_F(CSVTest, CSVWriter)
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col6(
    ones.begin(), ones.end(), numeric::scale_type{12});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col7(
    ones.begin(), ones.end(), numeric::scale_type{-12});

  std::vector<std::string> col8_data(num_rows, "rapids");
  cudf::test::strings_column_wrapper col8(col8_data.begin(), col8_data.end());

  cudf::table_view tab({col0, col1, col2, col3, col4, col5, col6, col7, col8});

  auto const filepath = temp_env->get_temp_dir() + "multicolumn.csv";
  auto w_options      = cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, tab)
                     .inter_column_delimiter(',');
  cudf::io::write_csv(w_options.build(), cudf::test::get_default_stream());
}

TEST_F(CSVTest, CSVReader)
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col6(
    ones.begin(), ones.end(), numeric::scale_type{12});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col7(
    ones.begin(), ones.end(), numeric::scale_type{-12});

  std::vector<std::string> col8_data(num_rows, "rapids");
  cudf::test::strings_column_wrapper col8(col8_data.begin(), col8_data.end());

  cudf::table_view tab({col0, col1, col2, col3, col4, col5, col6, col7, col8});

  auto const filepath = temp_env->get_temp_dir() + "multicolumn.csv";
  auto w_options      = cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, tab)
                     .inter_column_delimiter(',');
  cudf::io::write_csv(w_options.build(), cudf::test::get_default_stream());

  auto const r_options =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath}).build();
  cudf::io::read_csv(r_options, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
