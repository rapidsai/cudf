/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/io/api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/unary.hpp>

namespace cudf_io = cudf::io;

template <typename T, typename SourceElementT = T>
using column_wrapper =
  typename std::conditional<std::is_same<T, cudf::string_view>::value,
                            cudf::test::strings_column_wrapper,
                            cudf::test::fixed_width_column_wrapper<T, SourceElementT>>::type;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Base test fixture for tests
struct CsvReaderTest : public cudf::test::BaseFixture {
};

// Typed test fixture for timestamp type tests
template <typename T>
struct CsvReaderNumericTypeTest : public CsvReaderTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
using SupportedNumericTypes = cudf::test::Types<int64_t, double>;
TYPED_TEST_CASE(CsvReaderNumericTypeTest, SupportedNumericTypes);

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same<T1, bool>::value,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point<T1>::value,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

MATCHER_P(FloatNearPointwise, tolerance, "Out-of-range")
{
  return (std::get<0>(arg) > std::get<1>(arg) - tolerance &&
          std::get<0>(arg) < std::get<1>(arg) + tolerance);
}

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

// temporary method to verify the float columns until
// CUDF_TEST_EXPECT_COLUMNS_EQUAL supports floating point
template <typename T, typename valid_t>
void check_float_column(cudf::column_view const& col_lhs,
                        cudf::column_view const& col_rhs,
                        T tol,
                        valid_t const& validity)
{
  auto h_data = cudf::test::to_host<T>(col_rhs).first;

  std::vector<T> data(h_data.size());
  std::copy(h_data.begin(), h_data.end(), data.begin());

  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(col_lhs,
                                                (wrapper<T>{data.begin(), data.end(), validity}));
  CUDF_EXPECTS(col_lhs.null_count() == 0, "All elements should be valid");
  EXPECT_THAT(cudf::test::to_host<T>(col_lhs).first,
              ::testing::Pointwise(FloatNearPointwise(tol), data));
}

// timestamp column checker within tolerance
// given by `tol_ms` (miliseconds)
void check_timestamp_column(cudf::column_view const& col_lhs,
                            cudf::column_view const& col_rhs,
                            long tol_ms = 1000l)
{
  using T = cudf::timestamp_ms;
  using namespace cuda::std::chrono;

  auto h_lhs = cudf::test::to_host<T>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<T>(col_rhs).first;

  cudf::size_type nrows = h_lhs.size();
  EXPECT_TRUE(nrows == static_cast<cudf::size_type>(h_rhs.size()));

  auto begin_count = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end_count   = thrust::make_counting_iterator<cudf::size_type>(nrows);

  auto* ptr_lhs = h_lhs.data();  // cannot capture host_vector in thrust,
                                 // not even in host lambda
  auto* ptr_rhs = h_rhs.data();

  auto found = thrust::find_if(
    thrust::host, begin_count, end_count, [ptr_lhs, ptr_rhs, tol_ms](auto row_index) {
      auto delta_ms = cuda::std::chrono::duration_cast<cuda::std::chrono::milliseconds>(
        ptr_lhs[row_index] - ptr_rhs[row_index]);
      return delta_ms.count() >= tol_ms;
    });

  EXPECT_TRUE(found == end_count);  // not found...
}

// helper to replace in `str`  _all_ occurrences of `from` with `to`
std::string replace_all_helper(std::string str, const std::string& from, const std::string& to)
{
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

// compare string columns accounting for special character
// treatment: double double quotes ('\"')
// and surround whole string by double quotes if it contains:
// newline '\n', <delimiter>, and double quotes;
void check_string_column(cudf::column_view const& col_lhs,
                         cudf::column_view const& col_rhs,
                         std::string const& delimiter = ",")
{
  auto h_lhs = cudf::test::to_host<std::string>(col_lhs).first;
  auto h_rhs = cudf::test::to_host<std::string>(col_rhs).first;

  std::string newline("\n");
  std::string quotes("\"");
  std::string quotes_repl("\"\"");

  std::vector<std::string> v_lhs;
  std::transform(h_lhs.begin(),
                 h_lhs.end(),
                 std::back_inserter(v_lhs),
                 [delimiter, newline, quotes, quotes_repl](std::string const& str_row) {
                   auto found_quote = str_row.find(quotes);
                   auto found_newl  = str_row.find(newline);
                   auto found_delim = str_row.find(delimiter);

                   bool flag_found_quotes = (found_quote != std::string::npos);
                   bool need_surround = flag_found_quotes || (found_newl != std::string::npos) ||
                                        (found_delim != std::string::npos);

                   std::string str_repl;
                   if (flag_found_quotes) {
                     str_repl = replace_all_helper(str_row, quotes, quotes_repl);
                   } else {
                     str_repl = str_row;
                   }
                   return need_surround ? quotes + str_repl + quotes : str_row;
                 });
  EXPECT_TRUE(std::equal(v_lhs.begin(), v_lhs.end(), h_rhs.begin()));
}

// Helper function to compare two floating-point column contents
template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), lhs));
}

// Helper function to compare two column contents
template <typename T, typename std::enable_if_t<!std::is_floating_point<T>::value>* = nullptr>
void expect_column_data_equal(std::vector<T> const& lhs, cudf::column_view const& rhs)
{
  EXPECT_THAT(cudf::test::to_host<T>(rhs).first, ::testing::ElementsAreArray(lhs));
}

void write_csv_helper(std::string const& filename,
                      cudf::table_view const& table,
                      bool include_header,
                      std::vector<std::string> const& names = {})
{
  // csv_writer_options only keeps a pointer to metadata (non-owning)
  cudf_io::table_metadata metadata{};

  if (not names.empty()) {
    metadata.column_names = names;
  } else {
    // generate some dummy column names
    int i                  = 0;
    auto const num_columns = table.num_columns();
    metadata.column_names.reserve(num_columns);
    std::generate_n(std::back_inserter(metadata.column_names), num_columns, [&i]() {
      return std::string("col") + std::to_string(i++);
    });
  }

  cudf_io::csv_writer_options writer_options =
    cudf_io::csv_writer_options::builder(cudf_io::sink_info(filename), table)
      .include_header(include_header)
      .rows_per_chunk(
        1)  // Note: this gets adjusted to multiple of 8 (per legacy code logic and requirements)
      .metadata(&metadata);

  cudf_io::write_csv(writer_options);
}

template <typename T>
std::string assign(T input)
{
  return std::to_string(input);
}

std::string assign(std::string input) { return input; }

template <typename T>
std::vector<std::string> prepend_zeros(const std::vector<T>& input,
                                       int zero_count         = 0,
                                       bool add_positive_sign = false)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](const T& num) {
    auto str         = assign(num);
    bool is_negative = (str[0] == '-');
    if (is_negative) {
      str.insert(1, zero_count, '0');
      return str;
    } else if (add_positive_sign) {
      return "+" + std::string(zero_count, '0') + str;
    } else {
      str.insert(0, zero_count, '0');
      return str;
    }
  });
  return output;
}

}  // namespace

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumn)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });

  auto filepath = temp_env->get_temp_filepath("SingleColumn.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<TypeParam> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  expect_column_data_equal(std::vector<TypeParam>(sequence, sequence + num_rows), view.column(0));
}

TEST_F(CsvReaderTest, MultiColumn)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint8_values       = random_values<uint8_t>(num_rows);
  auto uint16_values      = random_values<uint16_t>(num_rows);
  auto uint32_values      = random_values<uint32_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int16_values[i]
           << "," << int32_values[i] << "," << int32_values[i] << "," << int64_values[i] << ","
           << int64_values[i] << "," << std::to_string(uint8_values[i]) << "," << uint16_values[i]
           << "," << uint32_values[i] << "," << uint64_values[i] << "," << float32_values[i] << ","
           << float32_values[i] << "," << float64_values[i] << "," << float64_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .header(-1)
      .dtypes({"int8",
               "short",
               "int16",
               "int",
               "int32",
               "long",
               "int64",
               "uint8",
               "uint16",
               "uint32",
               "uint64",
               "float",
               "float32",
               "double",
               "float64"});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int16_values, view.column(2));
  expect_column_data_equal(int32_values, view.column(3));
  expect_column_data_equal(int32_values, view.column(4));
  expect_column_data_equal(int64_values, view.column(5));
  expect_column_data_equal(int64_values, view.column(6));
  expect_column_data_equal(uint8_values, view.column(7));
  expect_column_data_equal(uint16_values, view.column(8));
  expect_column_data_equal(uint32_values, view.column(9));
  expect_column_data_equal(uint64_values, view.column(10));
  expect_column_data_equal(float32_values, view.column(11));
  expect_column_data_equal(float32_values, view.column(12));
  expect_column_data_equal(float64_values, view.column(13));
  expect_column_data_equal(float64_values, view.column(14));
}

TEST_F(CsvReaderTest, Booleans)
{
  auto filepath = temp_env->get_temp_dir() + "Booleans.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "YES,1,bar,true\nno,2,FOO,true\nBar,3,yes,false\nNo,4,NO,"
               "true\nYes,5,foo,false\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A", "B", "C", "D"})
      .dtypes({"int32", "int32", "short", "bool"})
      .true_values({"yes", "Yes", "YES", "foo", "FOO"})
      .false_values({"no", "No", "NO", "Bar", "bar"})
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  const auto view = result.tbl->view();
  EXPECT_EQ(4, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::INT32, view.column(1).type().id());
  ASSERT_EQ(cudf::type_id::INT16, view.column(2).type().id());
  ASSERT_EQ(cudf::type_id::BOOL8, view.column(3).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 0, 0, 0, 1}, view.column(0));
  expect_column_data_equal(std::vector<int16_t>{0, 1, 1, 0, 1}, view.column(2));
  expect_column_data_equal(std::vector<bool>{true, true, false, true, false}, view.column(3));
}

TEST_F(CsvReaderTest, Dates)
{
  auto filepath = temp_env->get_temp_dir() + "Dates.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampS.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1)
      .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_SECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_s>{cudf::timestamp_s{983750400s},
                                                          cudf::timestamp_s{1288483200s},
                                                          cudf::timestamp_s{782611200s},
                                                          cudf::timestamp_s{656208000s},
                                                          cudf::timestamp_s{0s},
                                                          cudf::timestamp_s{798163200s},
                                                          cudf::timestamp_s{774144000s},
                                                          cudf::timestamp_s{1149679230s},
                                                          cudf::timestamp_s{1126875750s},
                                                          cudf::timestamp_s{2764800s}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMilliSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampMs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1)
      .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MILLISECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_ms>{cudf::timestamp_ms{983750400000ms},
                                                           cudf::timestamp_ms{1288483200000ms},
                                                           cudf::timestamp_ms{782611200000ms},
                                                           cudf::timestamp_ms{656208000000ms},
                                                           cudf::timestamp_ms{0ms},
                                                           cudf::timestamp_ms{798163200000ms},
                                                           cudf::timestamp_ms{774144000000ms},
                                                           cudf::timestamp_ms{1149679230400ms},
                                                           cudf::timestamp_ms{1126875750400ms},
                                                           cudf::timestamp_ms{2764800000ms}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampMicroSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampUs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1)
      .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_MICROSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(std::vector<cudf::timestamp_us>{cudf::timestamp_us{983750400000000us},
                                                           cudf::timestamp_us{1288483200000000us},
                                                           cudf::timestamp_us{782611200000000us},
                                                           cudf::timestamp_us{656208000000000us},
                                                           cudf::timestamp_us{0us},
                                                           cudf::timestamp_us{798163200000000us},
                                                           cudf::timestamp_us{774144000000000us},
                                                           cudf::timestamp_us{1149679230400000us},
                                                           cudf::timestamp_us{1126875750400000us},
                                                           cudf::timestamp_us{2764800000000us}},
                           view.column(0));
}

TEST_F(CsvReaderTest, DatesCastToTimestampNanoSeconds)
{
  auto filepath = temp_env->get_temp_dir() + "DatesCastToTimestampNs.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "05/03/2001\n31/10/2010\n20/10/1994\n18/10/1990\n1/1/1970\n";
    outfile << "18/04/1995\n14/07/1994\n07/06/2006 11:20:30.400\n";
    outfile << "16/09/2005T1:2:30.400PM\n2/2/1970\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1)
      .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::TIMESTAMP_NANOSECONDS, view.column(0).type().id());

  using namespace cuda::std::chrono_literals;
  expect_column_data_equal(
    std::vector<cudf::timestamp_ns>{cudf::timestamp_ns{983750400000000000ns},
                                    cudf::timestamp_ns{1288483200000000000ns},
                                    cudf::timestamp_ns{782611200000000000ns},
                                    cudf::timestamp_ns{656208000000000000ns},
                                    cudf::timestamp_ns{0ns},
                                    cudf::timestamp_ns{798163200000000000ns},
                                    cudf::timestamp_ns{774144000000000000ns},
                                    cudf::timestamp_ns{1149679230400000000ns},
                                    cudf::timestamp_ns{1126875750400000000ns},
                                    cudf::timestamp_ns{2764800000000000ns}},
    view.column(0));
}

TEST_F(CsvReaderTest, FloatingPoint)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "5.6;0.5679e2;1.2e10;0.07e1;3000e-3;12.34e0;3.1e-001;-73."
               "98007199999998;";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"float32"})
      .lineterminator(';')
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::FLOAT32, view.column(0).type().id());

  const auto ref_vals =
    std::vector<float>{5.6, 56.79, 12000000000, 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  expect_column_data_equal(ref_vals, view.column(0));

  const auto bitmask = cudf::test::bitmask_to_host(view.column(0));
  ASSERT_EQ((1u << ref_vals.size()) - 1, bitmask[0]);
}

TEST_F(CsvReaderTest, Strings)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "Strings.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,abc def ghi" << '\n';
    outfile << "20,\"jkl mno pqr\"" << '\n';
    outfile << "30,stu \"\"vwx\"\" yz" << '\n';
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"})
      .quoting(cudf_io::quote_style::NONE);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc def ghi", "\"jkl mno pqr\"", "stu \"\"vwx\"\" yz"},
    view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotes)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotes.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,`abc,\ndef, ghi`" << '\n';
    outfile << "20,`jkl, ``mno``, pqr`" << '\n';
    outfile << "30,stu `vwx` yz" << '\n';
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"})
      .quotechar('`');
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"abc,\ndef, ghi", "jkl, `mno`, pqr", "stu `vwx` yz"}, view.column(1));
}

TEST_F(CsvReaderTest, StringsQuotesIgnored)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsQuotesIgnored.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << names[0] << ',' << names[1] << ',' << '\n';
    outfile << "10,\"abcdef ghi\"" << '\n';
    outfile << "20,\"jkl \"\"mno\"\" pqr\"" << '\n';
    outfile << "30,stu \"vwx\" yz" << '\n';
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"})
      .quoting(cudf_io::quote_style::NONE)
      .doublequote(false);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(2, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());
  ASSERT_EQ(cudf::type_id::STRING, view.column(1).type().id());

  expect_column_data_equal(
    std::vector<std::string>{"\"abcdef ghi\"", "\"jkl \"\"mno\"\" pqr\"", "stu \"vwx\" yz"},
    view.column(1));
}

TEST_F(CsvReaderTest, SkiprowsNrows)
{
  auto filepath = temp_env->get_temp_dir() + "SkiprowsNrows.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n2\n3\n4\n5\n6\n7\n8\n9\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"int32"})
      .header(1)
      .skiprows(2)
      .nrows(2);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{5, 6}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRange)
{
  auto filepath = temp_env->get_temp_dir() + "ByteRange.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1000\n2000\n3000\n4000\n5000\n6000\n7000\n8000\n9000\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"int32"})
      .header(-1)
      .byte_range_offset(11)
      .byte_range_size(15);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{4000, 5000, 6000}, view.column(0));
}

TEST_F(CsvReaderTest, ByteRangeStrings)
{
  std::string input = "\"a\"\n\"b\"\n\"c\"";
  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{input.c_str(), input.size()})
      .names({"A"})
      .dtypes({"str"})
      .header(-1)
      .byte_range_offset(4);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::STRING, view.column(0).type().id());

  expect_column_data_equal(std::vector<std::string>{"c"}, view.column(0));
}

TEST_F(CsvReaderTest, BlanksAndComments)
{
  auto filepath = temp_env->get_temp_dir() + "BlanksAndComments.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1\n#blank\n3\n4\n5\n#blank\n\n\n8\n9\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"int32"})
      .header(-1)
      .comment('#');
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT32, view.column(0).type().id());

  expect_column_data_equal(std::vector<int32_t>{1, 3, 4, 5, 8, 9}, view.column(0));
}

TEST_F(CsvReaderTest, EmptyFile)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, NoDataFile)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\n\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

TEST_F(CsvReaderTest, HeaderOnlyFile)
{
  auto filepath = temp_env->get_temp_dir() + "HeaderOnlyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "\"a\",\"b\",\"c\"\n\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_rows());
  EXPECT_EQ(3, view.num_columns());
}

TEST_F(CsvReaderTest, ArrowFileSource)
{
  auto filepath = temp_env->get_temp_dir() + "ArrowFileSource.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "A\n9\n8\n7\n6\n5\n4\n3\n2\n";
  }

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(filepath).Value(&infile).ok());

  auto arrow_source = cudf_io::arrow_io_source{infile};
  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{&arrow_source}).dtypes({"int8"});
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::INT8, view.column(0).type().id());

  expect_column_data_equal(std::vector<int8_t>{9, 8, 7, 6, 5, 4, 3, 2}, view.column(0));
}

TEST_F(CsvReaderTest, InvalidFloatingPoint)
{
  const auto filepath = temp_env->get_temp_dir() + "InvalidFloatingPoint.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "1.2e1+\n3.4e2-\n5.6e3e\n7.8e3A\n9.0Be1\n1C.2";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"float32"})
      .header(-1);
  const auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  EXPECT_EQ(1, view.num_columns());
  ASSERT_EQ(cudf::type_id::FLOAT32, view.column(0).type().id());

  const auto col_data = cudf::test::to_host<float>(view.column(0));
  // col_data.first contains the column data
  for (const auto& elem : col_data.first) ASSERT_TRUE(std::isnan(elem));
  // col_data.second contains the bitmasks
  ASSERT_EQ(0u, col_data.second[0]);
}

TEST_F(CsvReaderTest, StringInference)
{
  std::string buffer = "\"-1\"\n";
  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .header(-1);
  const auto result = cudf_io::read_csv(in_opts);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
}

TEST_F(CsvReaderTest, SkipRowsXorSkipFooter)
{
  std::string buffer = "1,2,3";

  cudf_io::csv_reader_options skiprows_options =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .header(-1)
      .skiprows(1);
  EXPECT_NO_THROW(cudf_io::read_csv(skiprows_options));

  cudf_io::csv_reader_options skipfooter_options =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .header(-1)
      .skipfooter(1);
  EXPECT_NO_THROW(cudf_io::read_csv(skipfooter_options));
}

TEST_F(CsvReaderTest, nullHandling)
{
  const auto filepath = temp_env->get_temp_dir() + "NullValues.csv";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "NULL\n\nnull\nn/a\nNull\nNA\nnan";
  }

  // Test disabling na_filter
  {
    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
        .na_filter(false)
        .dtypes({"str"})
        .header(-1)
        .skip_blank_lines(false);
    const auto result = cudf_io::read_csv(in_opts);
    const auto view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"});
    expect_columns_equal(expect, view.column(0));
  }

  // Test enabling na_filter
  {
    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
        .dtypes({"str"})
        .header(-1)
        .skip_blank_lines(false);
    const auto result = cudf_io::read_csv(in_opts);
    const auto view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, true, false, false});

    expect_columns_equal(expect, view.column(0));
  }

  // Setting na_values with default values
  {
    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
        .na_values({"Null"})
        .dtypes({"str"})
        .header(-1)
        .skip_blank_lines(false);
    const auto result = cudf_io::read_csv(in_opts);
    const auto view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {false, false, false, false, false, false, false});

    expect_columns_equal(expect, view.column(0));
  }

  // Setting na_values without default values
  {
    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
        .keep_default_na(false)
        .na_values({"Null"})
        .dtypes({"str"})
        .header(-1)
        .skip_blank_lines(false);
    const auto result = cudf_io::read_csv(in_opts);
    const auto view   = result.tbl->view();
    auto expect =
      cudf::test::strings_column_wrapper({"NULL", "", "null", "n/a", "Null", "NA", "nan"},
                                         {true, true, true, true, false, true, true, true});

    expect_columns_equal(expect, view.column(0));
  }
}

TEST_F(CsvReaderTest, FailCases)
{
  std::string buffer = "1,2,3";
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_offset(4)
        .skiprows(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_offset(4)
        .skipfooter(1),
      cudf::logic_error);
  }

  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_offset(4)
        .nrows(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_size(4)
        .skiprows(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_size(4)
        .skipfooter(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .byte_range_size(4)
        .nrows(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .skiprows(1)
        .byte_range_offset(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .skipfooter(1)
        .byte_range_offset(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .nrows(1)
        .byte_range_offset(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .skiprows(1)
        .byte_range_size(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .skipfooter(1)
        .byte_range_size(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .nrows(1)
        .byte_range_size(4),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .nrows(1)
        .skipfooter(1),
      cudf::logic_error);
    ;
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .skipfooter(1)
        .nrows(1),
      cudf::logic_error);
  }
  {
    EXPECT_THROW(
      cudf_io::csv_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
        .na_filter(false)
        .na_values({"Null"}),
      cudf::logic_error);
  }
}

TEST_F(CsvReaderTest, HexTest)
{
  auto filepath = temp_env->get_temp_filepath("Hexadecimal.csv");
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "0x0\n-0x1000\n0xfedcba\n0xABCDEF\n0xaBcDeF\n9512c20b\n";
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"hex"})
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  expect_column_data_equal(std::vector<int64_t>{0, -4096, 16702650, 11259375, 11259375, 2501034507},
                           result.tbl->view().column(0));
}

TYPED_TEST(CsvReaderNumericTypeTest, SingleColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return static_cast<TypeParam>(i + 1000.50f); });
  auto input_column = column_wrapper<TypeParam>(sequence, sequence + num_rows);
  auto input_table  = cudf::table_view{std::vector<cudf::column_view>{input_column}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithWriter.csv");

  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, MultiColumnWithWriter)
{
  constexpr auto num_rows = 10;
  auto int8_column        = []() {
    auto values = random_values<int8_t>(num_rows);
    return column_wrapper<int8_t>(values.begin(), values.end());
  }();
  auto int16_column = []() {
    auto values = random_values<int16_t>(num_rows);
    return column_wrapper<int16_t>(values.begin(), values.end());
  }();
  auto int32_column = []() {
    auto values = random_values<int32_t>(num_rows);
    return column_wrapper<int32_t>(values.begin(), values.end());
  }();
  auto int64_column = []() {
    auto values = random_values<int64_t>(num_rows);
    return column_wrapper<int64_t>(values.begin(), values.end());
  }();
  auto uint8_column = []() {
    auto values = random_values<uint8_t>(num_rows);
    return column_wrapper<uint8_t>(values.begin(), values.end());
  }();
  auto uint16_column = []() {
    auto values = random_values<uint16_t>(num_rows);
    return column_wrapper<uint16_t>(values.begin(), values.end());
  }();
  auto uint32_column = []() {
    auto values = random_values<uint32_t>(num_rows);
    return column_wrapper<uint32_t>(values.begin(), values.end());
  }();
  auto uint64_column = []() {
    auto values = random_values<uint64_t>(num_rows);
    return column_wrapper<uint64_t>(values.begin(), values.end());
  }();
  auto float32_column = []() {
    auto values = random_values<float>(num_rows);
    return column_wrapper<float>(values.begin(), values.end());
  }();
  auto float64_column = []() {
    auto values = random_values<double>(num_rows);
    return column_wrapper<double>(values.begin(), values.end());
  }();

  std::vector<cudf::column_view> input_columns{int8_column,
                                               int16_column,
                                               int16_column,
                                               int32_column,
                                               int32_column,
                                               int64_column,
                                               int64_column,
                                               uint8_column,
                                               uint16_column,
                                               uint32_column,
                                               uint64_column,
                                               float32_column,
                                               float32_column,
                                               float64_column,
                                               float64_column};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_dir() + "MultiColumnWithWriter.csv";

  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .header(-1)
      .dtypes({"int8",
               "short",
               "int16",
               "int",
               "int32",
               "long",
               "int64",
               "uint8",
               "uint16",
               "uint32",
               "uint64",
               "float",
               "float32",
               "double",
               "float64"});
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();

  std::vector<cudf::size_type> non_float64s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const auto input_sliced_view  = input_table.select(non_float64s);
  const auto result_sliced_view = result_table.select(non_float64s);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_sliced_view, result_sliced_view);

  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  double tol{1.0e-6};
  auto float64_col_idx = non_float64s.size();
  check_float_column(
    input_table.column(float64_col_idx), result_table.column(float64_col_idx), tol, validity);
  ++float64_col_idx;
  check_float_column(
    input_table.column(float64_col_idx), result_table.column(float64_col_idx), tol, validity);
}

TEST_F(CsvReaderTest, DatesWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DatesWithWriter.csv";

  auto input_column = column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{983750400000,
                                                                                  1288483200000,
                                                                                  782611200000,
                                                                                  656208000000,
                                                                                  0L,
                                                                                  798163200000,
                                                                                  774144000000,
                                                                                  1149679230400,
                                                                                  1126875750400,
                                                                                  2764800000};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO need to add a dayfirst flag?
  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"date"})
      .dayfirst(true)
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();

  check_timestamp_column(input_table.column(0), result_table.column(0));
}

TEST_F(CsvReaderTest, DatesStringWithWriter)
{
  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_D.csv";

    auto input_column = column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{-106751, 106751};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-22", "2262-04-11"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table, false);

    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).names({"A"}).header(-1);
    auto result = cudf_io::read_csv(in_opts);

    const auto result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_s.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{-9223372036, 9223372036};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:44Z", "2262-04-11T23:47:16Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table, false);

    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).names({"A"}).header(-1);
    auto result = cudf_io::read_csv(in_opts);

    const auto result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ms.csv";

    auto input_column =
      column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{-9223372036854, 9223372036854};
    auto expected_column =
      column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.146Z", "2262-04-11T23:47:16.854Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table, false);

    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).names({"A"}).header(-1);
    auto result = cudf_io::read_csv(in_opts);

    const auto result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_us.csv";

    auto input_column = column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>{
      -9223372036854775, 9223372036854775};
    auto cast_column     = cudf::strings::from_timestamps(input_column, "%Y-%m-%dT%H:%M:%S.%fZ");
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145225Z",
                                                             "2262-04-11T23:47:16.854775Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table, false);

    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).names({"A"}).header(-1);
    auto result = cudf_io::read_csv(in_opts);

    const auto result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }

  {
    auto filepath = temp_env->get_temp_dir() + "DatesStringWithWriter_ns.csv";

    auto input_column = column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>{
      -9223372036854775807, 9223372036854775807};
    auto expected_column = column_wrapper<cudf::string_view>{"1677-09-21T00:12:43.145224193Z",
                                                             "2262-04-11T23:47:16.854775807Z"};

    cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

    write_csv_helper(filepath, input_table, false);

    cudf_io::csv_reader_options in_opts =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).names({"A"}).header(-1);
    auto result = cudf_io::read_csv(in_opts);

    const auto result_table = result.tbl->view();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column, result_table.column(0));
  }
}

TEST_F(CsvReaderTest, FloatingPointWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPointWithWriter.csv";

  auto input_column =
    column_wrapper<double>{5.6, 56.79, 12000000000., 0.7, 3.000, 12.34, 0.31, -73.98007199999998};
  cudf::table_view input_table(std::vector<cudf::column_view>{input_column});

  // TODO add lineterminator=";"
  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names({"A"})
      .dtypes({"float64"})
      .header(-1);
  // in_opts.lineterminator = ';';
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, StringsWithWriter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriter.csv";

  auto int_column = column_wrapper<int32_t>{10, 20, 30};
  auto string_column =
    column_wrapper<cudf::string_view>{"abc def ghi", "\"jkl mno pqr\"", "stu \"\"vwx\"\" yz"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, true, names);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"})
      .quoting(cudf_io::quote_style::NONE);
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
}

TEST_F(CsvReaderTest, StringsWithWriterSimple)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl mno pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  // TODO add quoting style flag?
  write_csv_helper(filepath, input_table, true, names);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"})
      .quoting(cudf_io::quote_style::NONE);
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_table.column(0), result_table.column(0));
  check_string_column(input_table.column(1), result_table.column(1));
}

TEST_F(CsvReaderTest, StringsEmbeddedDelimiter)
{
  std::vector<std::string> names{"line", "verse"};

  auto filepath = temp_env->get_temp_dir() + "StringsWithWriterSimple.csv";

  auto int_column    = column_wrapper<int32_t>{10, 20, 30};
  auto string_column = column_wrapper<cudf::string_view>{"abc def ghi", "jkl,mno,pq", "stu vwx y"};
  cudf::table_view input_table(std::vector<cudf::column_view>{int_column, string_column});

  write_csv_helper(filepath, input_table, true, names);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"int32", "str"});
  auto result = cudf_io::read_csv(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result.tbl->view());
}

TEST_F(CsvReaderTest, EmptyFileWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFileWithWriter.csv";

  cudf::table_view empty_table;

  // TODO is it ok for write_csv to throw instead of just writing an empty file?
  EXPECT_THROW(write_csv_helper(filepath, empty_table, false), cudf::logic_error);
}

class TestSource : public cudf::io::datasource {
 public:
  std::string const str;

  TestSource(std::string s) : str(std::move(s)) {}
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    size = std::min(size, str.size() - offset);
    return std::make_unique<non_owning_buffer>((uint8_t*)str.data() + offset, size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, str.size() - offset);
    memcpy(dst, str.data() + offset, size);
    return read_size;
  }

  size_t size() const override { return str.size(); }
};

TEST_F(CsvReaderTest, UserImplementedSource)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);

  std::ostringstream csv_data;
  for (int i = 0; i < num_rows; ++i) {
    csv_data << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
             << "\n";
  }
  TestSource source{csv_data.str()};
  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{&source})
      .dtypes({"int8", "int16", "int32"})
      .header(-1);
  auto result = cudf_io::read_csv(in_opts);

  auto const view = result.tbl->view();
  expect_column_data_equal(int8_values, view.column(0));
  expect_column_data_equal(int16_values, view.column(1));
  expect_column_data_equal(int32_values, view.column(2));
}

TEST_F(CsvReaderTest, DurationsWithWriter)
{
  auto filepath = temp_env->get_temp_dir() + "DurationsWithWriter.csv";

  constexpr long max_value_d  = std::numeric_limits<cudf::duration_D::rep>::max();
  constexpr long min_value_d  = std::numeric_limits<cudf::duration_D::rep>::min();
  constexpr long max_value_ns = std::numeric_limits<cudf::duration_s::rep>::max();
  constexpr long min_value_ns = std::numeric_limits<cudf::duration_s::rep>::min();
  column_wrapper<cudf::duration_D, cudf::duration_D::rep> durations_D{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_d, max_value_d}};
  column_wrapper<cudf::duration_s, int64_t> durations_s{{-86400L,
                                                         -3600L,
                                                         -2L,
                                                         -1L,
                                                         0L,
                                                         1L,
                                                         2L,
                                                         min_value_ns / 1000000000 + 1,
                                                         max_value_ns / 1000000000}};
  column_wrapper<cudf::duration_ms, int64_t> durations_ms{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000000 + 1, max_value_ns / 1000000}};
  column_wrapper<cudf::duration_us, int64_t> durations_us{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns / 1000 + 1, max_value_ns / 1000}};
  column_wrapper<cudf::duration_ns, int64_t> durations_ns{
    {-86400L, -3600L, -2L, -1L, 0L, 1L, 2L, min_value_ns, max_value_ns}};

  cudf::table_view input_table(std::vector<cudf::column_view>{
    durations_D, durations_s, durations_ms, durations_us, durations_ns});
  std::vector<std::string> names{"D", "s", "ms", "us", "ns"};

  write_csv_helper(filepath, input_table, true, names);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath})
      .names(names)
      .dtypes({"timedelta[D]",
               "timedelta64[s]",
               "timedelta64[ms]",
               "timedelta64[us]",
               "timedelta64[ns]"});
  auto result = cudf_io::read_csv(in_opts);

  const auto result_table = result.tbl->view();
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, result_table);
}

TEST_F(CsvReaderTest, ParseInRangeIntegers)
{
  std::vector<int64_t> small_int               = {0, -10, 20, -30};
  std::vector<int64_t> less_equal_int64_max    = {std::numeric_limits<int64_t>::max() - 3,
                                               std::numeric_limits<int64_t>::max() - 2,
                                               std::numeric_limits<int64_t>::max() - 1,
                                               std::numeric_limits<int64_t>::max()};
  std::vector<int64_t> greater_equal_int64_min = {std::numeric_limits<int64_t>::min() + 3,
                                                  std::numeric_limits<int64_t>::min() + 2,
                                                  std::numeric_limits<int64_t>::min() + 1,
                                                  std::numeric_limits<int64_t>::min()};
  std::vector<uint64_t> greater_int64_max      = {uint64_t{std::numeric_limits<int64_t>::max()} - 1,
                                             uint64_t{std::numeric_limits<int64_t>::max()},
                                             uint64_t{std::numeric_limits<int64_t>::max()} + 1,
                                             uint64_t{std::numeric_limits<int64_t>::max()} + 2};
  std::vector<uint64_t> less_equal_uint64_max  = {std::numeric_limits<uint64_t>::max() - 3,
                                                 std::numeric_limits<uint64_t>::max() - 2,
                                                 std::numeric_limits<uint64_t>::max() - 1,
                                                 std::numeric_limits<uint64_t>::max()};
  auto input_small_int = column_wrapper<int64_t>(small_int.begin(), small_int.end());
  auto input_less_equal_int64_max =
    column_wrapper<int64_t>(less_equal_int64_max.begin(), less_equal_int64_max.end());
  auto input_greater_equal_int64_min =
    column_wrapper<int64_t>(greater_equal_int64_min.begin(), greater_equal_int64_min.end());
  auto input_greater_int64_max =
    column_wrapper<uint64_t>(greater_int64_max.begin(), greater_int64_max.end());
  auto input_less_equal_uint64_max =
    column_wrapper<uint64_t>(less_equal_uint64_max.begin(), less_equal_uint64_max.end());

  auto small_int_append_zeros               = prepend_zeros(small_int, 32, true);
  auto less_equal_int64_max_append_zeros    = prepend_zeros(less_equal_int64_max, 32, true);
  auto greater_equal_int64_min_append_zeros = prepend_zeros(greater_equal_int64_min, 17);
  auto greater_int64_max_append_zeros       = prepend_zeros(greater_int64_max, 5);
  auto less_equal_uint64_max_append_zeros   = prepend_zeros(less_equal_uint64_max, 8, true);

  auto input_small_int_append =
    column_wrapper<cudf::string_view>(small_int_append_zeros.begin(), small_int_append_zeros.end());
  auto input_less_equal_int64_max_append = column_wrapper<cudf::string_view>(
    less_equal_int64_max_append_zeros.begin(), less_equal_int64_max_append_zeros.end());
  auto input_greater_equal_int64_min_append = column_wrapper<cudf::string_view>(
    greater_equal_int64_min_append_zeros.begin(), greater_equal_int64_min_append_zeros.end());
  auto input_greater_int64_max_append = column_wrapper<cudf::string_view>(
    greater_int64_max_append_zeros.begin(), greater_int64_max_append_zeros.end());
  auto input_less_equal_uint64_max_append = column_wrapper<cudf::string_view>(
    less_equal_uint64_max_append_zeros.begin(), less_equal_uint64_max_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_small_int,
                                               input_less_equal_int64_max,
                                               input_greater_equal_int64_min,
                                               input_greater_int64_max,
                                               input_less_equal_uint64_max,
                                               input_small_int_append,
                                               input_less_equal_int64_max_append,
                                               input_greater_equal_int64_min_append,
                                               input_greater_int64_max_append,
                                               input_less_equal_uint64_max_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseInRangeIntegers.csv");

  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(4));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_small_int, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_int64_max, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_equal_int64_min, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_int64_max, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_equal_uint64_max, view.column(9));
}

TEST_F(CsvReaderTest, ParseOutOfRangeIntegers)
{
  std::vector<std::string> out_of_range_positive = {"111111111111111111111",
                                                    "2222222222222222222222",
                                                    "33333333333333333333333",
                                                    "444444444444444444444444"};
  std::vector<std::string> out_of_range_negative = {"-111111111111111111111",
                                                    "-2222222222222222222222",
                                                    "-33333333333333333333333",
                                                    "-444444444444444444444444"};
  std::vector<std::string> greater_uint64_max    = {
    "18446744073709551615", "18446744073709551616", "18446744073709551617", "18446744073709551618"};
  std::vector<std::string> less_int64_min = {
    "-9223372036854775807", "-9223372036854775808", "-9223372036854775809", "-9223372036854775810"};
  std::vector<std::string> mixed_range = {
    "18446744073709551613", "18446744073709551614", "18446744073709551615", "-5"};
  auto input_out_of_range_positive =
    column_wrapper<cudf::string_view>(out_of_range_positive.begin(), out_of_range_positive.end());
  auto input_out_of_range_negative =
    column_wrapper<cudf::string_view>(out_of_range_negative.begin(), out_of_range_negative.end());
  auto input_greater_uint64_max =
    column_wrapper<cudf::string_view>(greater_uint64_max.begin(), greater_uint64_max.end());
  auto input_less_int64_min =
    column_wrapper<cudf::string_view>(less_int64_min.begin(), less_int64_min.end());
  auto input_mixed_range =
    column_wrapper<cudf::string_view>(mixed_range.begin(), mixed_range.end());

  auto out_of_range_positive_append_zeros = prepend_zeros(out_of_range_positive, 32, true);
  auto out_of_range_negative_append_zeros = prepend_zeros(out_of_range_negative, 5);
  auto greater_uint64_max_append_zeros    = prepend_zeros(greater_uint64_max, 8, true);
  auto less_int64_min_append_zeros        = prepend_zeros(less_int64_min, 17);
  auto mixed_range_append_zeros           = prepend_zeros(mixed_range, 2, true);

  auto input_out_of_range_positive_append = column_wrapper<cudf::string_view>(
    out_of_range_positive_append_zeros.begin(), out_of_range_positive_append_zeros.end());
  auto input_out_of_range_negative_append = column_wrapper<cudf::string_view>(
    out_of_range_negative_append_zeros.begin(), out_of_range_negative_append_zeros.end());
  auto input_greater_uint64_max_append = column_wrapper<cudf::string_view>(
    greater_uint64_max_append_zeros.begin(), greater_uint64_max_append_zeros.end());
  auto input_less_int64_min_append = column_wrapper<cudf::string_view>(
    less_int64_min_append_zeros.begin(), less_int64_min_append_zeros.end());
  auto input_mixed_range_append = column_wrapper<cudf::string_view>(
    mixed_range_append_zeros.begin(), mixed_range_append_zeros.end());

  std::vector<cudf::column_view> input_columns{input_out_of_range_positive,
                                               input_out_of_range_negative,
                                               input_greater_uint64_max,
                                               input_less_int64_min,
                                               input_mixed_range,
                                               input_out_of_range_positive_append,
                                               input_out_of_range_negative_append,
                                               input_greater_uint64_max_append,
                                               input_less_int64_min_append,
                                               input_mixed_range_append};
  cudf::table_view input_table{input_columns};

  auto filepath = temp_env->get_temp_filepath("ParseOutOfRangeIntegers.csv");

  write_csv_helper(filepath, input_table, false);

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive, view.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative, view.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max, view.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min, view.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range, view.column(4));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_positive_append, view.column(5));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_out_of_range_negative_append, view.column(6));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_greater_uint64_max_append, view.column(7));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_less_int64_min_append, view.column(8));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input_mixed_range_append, view.column(9));
}

TEST_F(CsvReaderTest, ReadMaxNumericValue)
{
  constexpr auto num_rows = 10;
  auto sequence           = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return std::numeric_limits<uint64_t>::max() - i; });

  auto filepath = temp_env->get_temp_filepath("ReadMaxNumericValue.csv");
  {
    std::ofstream out_file{filepath, std::ofstream::out};
    std::ostream_iterator<uint64_t> output_iterator(out_file, "\n");
    std::copy(sequence, sequence + num_rows, output_iterator);
  }

  cudf_io::csv_reader_options in_opts =
    cudf_io::csv_reader_options::builder(cudf_io::source_info{filepath}).header(-1);
  auto result = cudf_io::read_csv(in_opts);

  const auto view = result.tbl->view();
  expect_column_data_equal(std::vector<uint64_t>(sequence, sequence + num_rows), view.column(0));
}

CUDF_TEST_PROGRAM_MAIN()
