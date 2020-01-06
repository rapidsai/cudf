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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/io/functions.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/io/api.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace cudf_io = cudf::experimental::io;

template <typename T>
using column_wrapper =
    typename std::conditional<std::is_same<T, cudf::string_view>::value,
                              cudf::test::strings_column_wrapper,
                              cudf::test::fixed_width_column_wrapper<T>>::type;
using column = cudf::column;
using table = cudf::experimental::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(
        new cudf::test::TempDirTestEnvironment));

// Base test fixture for tests
struct CsvReaderTest : public cudf::test::BaseFixture {};

// Typed test fixture for timestamp type tests
template <typename T>
struct CsvReaderNumericTypeTest : public CsvReaderTest {
  auto type() { return cudf::data_type{cudf::experimental::type_to_id<T>()}; }
};

// Declare typed test cases
using SupportedNumericTypes = cudf::test::Types<int64_t, double>;
TYPED_TEST_CASE(CsvReaderNumericTypeTest, SupportedNumericTypes);

namespace {

// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size) {
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution = typename std::conditional_t<
      std::is_same<T1, bool>::value, std::bernoulli_distribution,
      std::conditional_t<std::is_floating_point<T1>::value,
                         std::uniform_real_distribution<T1>,
                         std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

MATCHER_P(FloatNearPointwise, tolerance, "Out-of-range") {
  return (std::get<0>(arg) > std::get<1>(arg) - tolerance &&
          std::get<0>(arg) < std::get<1>(arg) + tolerance);
}

// Helper function to compare two floating-point column contents
template <typename T, typename std::enable_if_t<
                          std::is_floating_point<T>::value>* = nullptr>
void expect_column_data_equal(std::vector<T> const& lhs,
                              cudf::column_view const& rhs) {
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), lhs));
}

// Helper function to compare two column contents
template <typename T, typename std::enable_if_t<
                          !std::is_floating_point<T>::value>* = nullptr>
void expect_column_data_equal(std::vector<T> const& lhs,
                              cudf::column_view const& rhs) {
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first,
              ::testing::ElementsAreArray(lhs));
}

}  // namespace

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumn) {
  constexpr auto num_rows = 10;
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });

  auto filepath = temp_env->get_temp_filepath("SingleColumn.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<TypeParam> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.header = -1;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  expect_column_data_equal(
      std::vector<TypeParam>(sequence, sequence + num_rows), view.column(0));
}

TEST_F(CsvReaderTest, MultiColumn) {
  constexpr auto num_rows = 10;
  auto int8_values = random_values<int8_t>(num_rows);
  auto int16_values = random_values<int16_t>(num_rows);
  auto int32_values = random_values<int32_t>(num_rows);
  auto int64_values = random_values<int64_t>(num_rows);
  auto float32_values = random_values<float>(num_rows);
  auto float64_values = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << "," << int16_values[i] << ","
           << int16_values[i] << "," << int32_values[i] << ","
           << int32_values[i] << "," << int64_values[i] << ","
           << int64_values[i] << "," << float32_values[i] << ","
           << float32_values[i] << "," << float64_values[i] << ","
           << float64_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.dtype = {"int8",  "short", "int16",   "int",    "int32",  "long",
                   "int64", "float", "float32", "double", "float64"};
  in_args.header = -1;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int16_values, view.column(2));
  expect_column_data_equal(int32_values, view.column(3));
  expect_column_data_equal(int32_values, view.column(4));
  expect_column_data_equal(int64_values, view.column(5));
  expect_column_data_equal(int64_values, view.column(6));
  expect_column_data_equal(float32_values, view.column(7));
  expect_column_data_equal(float32_values, view.column(8));
  expect_column_data_equal(float64_values, view.column(9));
  expect_column_data_equal(float64_values, view.column(10));
}

TEST_F(CsvReaderTest, Booleans) {
  auto filepath = temp_env->get_temp_dir() + "Booleans.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
               "true\nYes,5,foo,false\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A", "B", "C", "D"};
  in_args.dtype = {"int32", "int32", "short", "bool"};
  in_args.true_values = {"yes", "Yes", "YES", "foo", "FOO"};
  in_args.false_values = {"no", "No", "NO", "Bar", "bar"};
  in_args.header = -1;
  auto result = cudf_io::read_csv(in_args);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  const auto view = result.tbl->view();
  EXPECT_EQ(4, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::INT32, view.column(1).type().id());
  ASSERT_EQ(cudf::type_id::INT16, view.column(2).type().id());
  ASSERT_EQ(cudf::type_id::BOOL8, view.column(3).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 0, 0, 0, 1}, view.column(0));
  expect_column_data_equal(std::vector<int16_t>{0, 1, 1, 0, 1}, view.column(2));
  expect_column_data_equal(
      std::vector<cudf::experimental::bool8>{
          cudf::experimental::true_v, cudf::experimental::true_v,
          cudf::experimental::false_v, cudf::experimental::true_v,
          cudf::experimental::false_v},
      view.column(3));
}

TEST_F(CsvReaderTest, Dates) {
  auto filepath = temp_env->get_temp_dir() + "Dates.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"date"};
  in_args.dayfirst = true;
  in_args.header = -1;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<cudf::timestamp_ms>{
          983750400000, 1288483200000, 782611200000, 656208000000, 0,
          798163200000, 774144000000, 1149679230400, 1126875750400, 2764800000},
      view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampSeconds) {
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampS.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"date"};
  in_args.dayfirst = true;
  in_args.header = -1;
  in_args.timestamp_type = cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<cudf::timestamp_s>{983750400, 1288483200, 782611200,
                                     656208000, 0, 798163200, 774144000,
                                     1149679230, 1126875750, 2764800},
      view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMilliSeconds) {
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampMs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"date"};
  in_args.dayfirst = true;
  in_args.header = -1;
  in_args.timestamp_type =
      cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<cudf::timestamp_ms>{
          983750400000, 1288483200000, 782611200000, 656208000000, 0,
          798163200000, 774144000000, 1149679230400, 1126875750400, 2764800000},
      view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMicroSeconds) {
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampUs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"date"};
  in_args.dayfirst = true;
  in_args.header = -1;
  in_args.timestamp_type =
      cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<cudf::timestamp_us>{
          983750400000000, 1288483200000000, 782611200000000, 656208000000000,
          0, 798163200000000, 774144000000000, 1149679230400000,
          1126875750400000, 2764800000000},
      view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampNanoSeconds) {
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampNs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"date"};
  in_args.dayfirst = true;
  in_args.header = -1;
  in_args.timestamp_type =
      cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<cudf::timestamp_ns>{
          983750400000000000, 1288483200000000000, 782611200000000000,
          656208000000000000, 0, 798163200000000000, 774144000000000000,
          1149679230400000000, 1126875750400000000, 2764800000000000},
      view.column(0));
}

TEST_F(CsvReaderTest, FloatingPoint) {
  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73."
               "98007199999998;";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"float32"};
  in_args.lineterminator = ';';
  in_args.header = -1;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::FLOAT32, view.column(0).type().id());

  expect_column_data_equal(
      std::vector<float>{5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31,
                         -73.98007199999998},
      view.column(0));
}

TEST_F(CsvReaderTest, Strings) {
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "Strings.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << "30,stu \"\"vwx\"\" yz" << '\n';
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = names;
  in_args.dtype = {"int32", "str"};
  in_args.quoting = cudf_io::quote_style::NONE;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
      std::vector<std::string>{"abc def ghi", "\"jkl mno pqr\"",
                               "stu \"\"vwx\"\" yz"},
      view.column(1));
}

TEST_F(CsvReaderTest, DISABLED_StringsQuotes) {
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotes.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = names;
  in_args.dtype = {"int32", "str"};
  in_args.quotechar = '`';
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
      std::vector<std::string>{"abc,\ndef, ghi", "jkl, `mno`, pqr",
                               "stu `vwx` yz"},
      view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesIgnored) {
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << "20,\"jkl \"\"mno\"\" pqr\"" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = names;
  in_args.dtype = {"int32", "str"};
  in_args.quoting = cudf_io::quote_style::NONE;
  in_args.doublequote = false;  // do not replace double quotechar with single
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
      std::vector<std::string>{"\"abcdef ghi\"", "\"jkl \"\"mno\"\" pqr\"",
                               "stu \"vwx\" yz"},
      view.column(1));
}

TEST_F(CsvReaderTest, SkiprowsNrows) {
  auto filepath = temp_env->get_temp_dir() + "SkiprowsNrows.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"int32"};
  in_args.header = 1;
  in_args.skiprows = 2;
  in_args.skipfooter = 0;
  in_args.nrows = 2;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{5, 6}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRange) {
  auto filepath = temp_env->get_temp_dir() + "ByteRange.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"int32"};
  in_args.header = -1;
  in_args.byte_range_offset = 11;
  in_args.byte_range_size = 15;
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{4000, 5000, 6000},
                           view.column(0));
}

TEST_F(CsvReaderTest, BlanksAndComments) {
  auto filepath = temp_env->get_temp_dir() + "BlanksAndComments.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  in_args.names = {"A"};
  in_args.dtype = {"int32"};
  in_args.header = -1;
  in_args.comment = '#';
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 3, 4, 5, 8, 9},
                           view.column(0));
}

TEST_F(CsvReaderTest, EmptyFile) {
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, NoDataFile) {
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\n\n";
  }

  cudf_io::read_csv_args in_args{cudf_io::source_info{filepath}};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, ArrowFileSource) {
  auto filepath = temp_env->get_temp_dir() + "ArrowFileSource.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "A\n9\n8\n7\n6\n5\n4\n3\n2\n";
  }

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(filepath, &infile).ok());

  cudf_io::read_csv_args in_args{cudf_io::source_info{infile}};
  in_args.dtype = {"int8"};
  auto result = cudf_io::read_csv(in_args);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT8, view.column(0).type().id());

  expect_column_data_equal(std::vector<int8_t>{9, 8, 7, 6, 5, 4, 3, 2},
                           view.column(0));
}
