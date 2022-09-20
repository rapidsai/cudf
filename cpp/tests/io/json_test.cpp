/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/io/api.h>

#include <fstream>
#include <type_traits>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper        = wrapper<float>;
using float64_wrapper      = wrapper<double>;
using int_wrapper          = wrapper<int>;
using int8_wrapper         = wrapper<int8_t>;
using int16_wrapper        = wrapper<int16_t>;
using int64_wrapper        = wrapper<int64_t>;
using timestamp_ms_wrapper = wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>;
using bool_wrapper         = wrapper<bool>;

using cudf::data_type;
using cudf::type_id;
using cudf::type_to_id;

template <typename T>
auto dtype()
{
  return data_type{type_to_id<T>()};
}

template <typename T, typename SourceElementT = T>
using column_wrapper =
  typename std::conditional<std::is_same_v<T, cudf::string_view>,
                            cudf::test::strings_column_wrapper,
                            cudf::test::fixed_width_column_wrapper<T, SourceElementT>>::type;

namespace cudf_io = cudf::io;

cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T>
std::vector<std::string> prepend_zeros(const std::vector<T>& input,
                                       int zero_count         = 0,
                                       bool add_positive_sign = false)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](const T& num) {
    auto str         = std::to_string(num);
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

template <>
std::vector<std::string> prepend_zeros<std::string>(const std::vector<std::string>& input,
                                                    int zero_count,
                                                    bool add_positive_sign)
{
  std::vector<std::string> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [=](const std::string& num) {
    auto str         = num;
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

// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
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

// temporary method to verify the float columns until
// CUDF_TEST_EXPECT_COLUMNS_EQUAL supports floating point
template <typename T, typename valid_t>
void check_float_column(cudf::column_view const& col,
                        std::vector<T> const& data,
                        valid_t const& validity)
{
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(col, (wrapper<T>{data.begin(), data.end(), validity}));
  CUDF_EXPECTS(col.null_count() == 0, "All elements should be valid");
  EXPECT_THAT(cudf::test::to_host<T>(col).first,
              ::testing::Pointwise(FloatNearPointwise(1e-6), data));
}

/**
 * @brief Base test fixture for JSON reader tests
 */
struct JsonReaderTest : public cudf::test::BaseFixture {
};

TEST_F(JsonReaderTest, BasicJsonLines)
{
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<double>()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2, 3}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 float64_wrapper{{1.1, 2.2, 3.3}, validity});
}

TEST_F(JsonReaderTest, FloatingPoint)
{
  auto filepath = temp_env->get_temp_dir() + "FloatingPoint.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "[5.6]\n[0.5679e2]\n[1.2e10]\n[0.07e1]\n[3000e-3]\n[12.34e0]\n[3.1e-001]\n[-73."
               "98007199999998]\n";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({dtype<float>()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT32);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    float_wrapper{{5.6, 56.79, 12000000000., 0.7, 3.000, 12.34, 0.31, -73.98007199999998},
                  validity});

  const auto bitmask = cudf::test::bitmask_to_host(result.tbl->get_column(0));
  ASSERT_EQ((1u << result.tbl->get_column(0).size()) - 1, bitmask[0]);
}

TEST_F(JsonReaderTest, JsonLinesStrings)
{
  std::string data = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
      .dtypes({{"2", dtype<cudf::string_view>()}, {"0", dtype<int32_t>()}, {"1", dtype<double>()}})
      .lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  EXPECT_EQ(result.metadata.schema_info[2].name, "2");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_F(JsonReaderTest, MultiColumn)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.json";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << "[" << std::to_string(int8_values[i]) << "," << int16_values[i] << ","
           << int32_values[i] << "," << int64_values[i] << "," << float32_values[i] << ","
           << float64_values[i] << "]\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({dtype<int8_t>(),
               dtype<int16_t>(),
               dtype<int32_t>(),
               dtype<int64_t>(),
               dtype<float>(),
               dtype<double>()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  const auto view = result.tbl->view();

  EXPECT_EQ(view.column(0).type().id(), cudf::type_id::INT8);
  EXPECT_EQ(view.column(1).type().id(), cudf::type_id::INT16);
  EXPECT_EQ(view.column(2).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(view.column(3).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(view.column(4).type().id(), cudf::type_id::FLOAT32);
  EXPECT_EQ(view.column(5).type().id(), cudf::type_id::FLOAT64);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0),
                                 int8_wrapper{int8_values.begin(), int8_values.end(), validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1),
                                 int16_wrapper{int16_values.begin(), int16_values.end(), validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(2),
                                 int_wrapper{int32_values.begin(), int32_values.end(), validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(3),
                                 int64_wrapper{int64_values.begin(), int64_values.end(), validity});
  check_float_column(view.column(4), float32_values, validity);
  check_float_column(view.column(5), float64_values, validity);
}

TEST_F(JsonReaderTest, Booleans)
{
  auto filepath = temp_env->get_temp_dir() + "Booleans.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "[true]\n[true]\n[false]\n[false]\n[true]";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({dtype<bool>()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  // Booleans are the same (integer) data type, but valued at 0 or 1
  const auto view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::BOOL8);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 bool_wrapper{{true, true, false, false, true}, validity});
}

TEST_F(JsonReaderTest, Dates)
{
  auto filepath = temp_env->get_temp_dir() + "Dates.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "[05/03/2001]\n[31/10/2010]\n[20/10/1994]\n[18/10/1990]\n[1/1/1970]\n";
    outfile << "[18/04/1995]\n[14/07/1994]\n[07/06/2006 11:20:30.400]\n";
    outfile << "[16/09/2005T1:2:30.400PM]\n[2/2/1970]\n";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({data_type{type_id::TIMESTAMP_MILLISECONDS}})
      .lines(true)
      .dayfirst(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  const auto view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::TIMESTAMP_MILLISECONDS);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 timestamp_ms_wrapper{{983750400000,
                                                       1288483200000,
                                                       782611200000,
                                                       656208000000,
                                                       0L,
                                                       798163200000,
                                                       774144000000,
                                                       1149679230400,
                                                       1126875750400,
                                                       2764800000},
                                                      validity});
}

TEST_F(JsonReaderTest, Durations)
{
  auto filepath = temp_env->get_temp_dir() + "Durations.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "[-2]\n[-1]\n[0]\n";
    outfile << "[1 days]\n[0 days 23:01:00]\n[0 days 00:00:00.000000123]\n";
    outfile << "[0:0:0.000123]\n[0:0:0.000123000]\n[00:00:00.100000001]\n";
    outfile << "[-2147483648]\n[2147483647]\n";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({data_type{type_id::DURATION_NANOSECONDS}})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  const auto view = result.tbl->view();
  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::DURATION_NANOSECONDS);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(0),
    wrapper<cudf::duration_ns, cudf::duration_ns::rep>{{-2L,
                                                        -1L,
                                                        0L,
                                                        1L * 60 * 60 * 24 * 1000000000L,
                                                        (23 * 60 + 1) * 60 * 1000000000L,
                                                        123L,
                                                        123000L,
                                                        123000L,
                                                        100000001L,
                                                        -2147483648L,
                                                        2147483647L},
                                                       validity});
}

TEST_F(JsonReaderTest, JsonLinesDtypeInference)
{
  std::string data = "[100, 1.1, \"aa \"]\n[200, 2.2, \"  bbb\"]";

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
      .lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");
  EXPECT_EQ(result.metadata.schema_info[2].name, "2");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_F(JsonReaderTest, JsonLinesFileInput)
{
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesFileTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[11, 1.1]\n[22, 2.2]";
  outfile.close();

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{fname}).lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{11, 22}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});
}

TEST_F(JsonReaderTest, JsonLinesByteRange)
{
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesByteRangeTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]\n";
  outfile.close();

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{fname})
      .lines(true)
      .byte_range_offset(11)
      .byte_range_size(20);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "0");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{3000, 4000, 5000}, validity});
}

TEST_F(JsonReaderTest, JsonLinesObjects)
{
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesObjectsTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << " {\"co\\\"l1\" : 1, \"col2\" : 2.0} \n";
  outfile.close();

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{fname}).lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 1);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.metadata.schema_info[0].name, "co\\\"l1");
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.metadata.schema_info[1].name, "col2");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{1}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{2.0}, validity});
}

TEST_F(JsonReaderTest, JsonLinesObjectsStrings)
{
  auto test_json_objects = [](std::string const& data) {
    cudf_io::json_reader_options in_options =
      cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
        .lines(true);

    cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

    EXPECT_EQ(result.tbl->num_columns(), 3);
    EXPECT_EQ(result.tbl->num_rows(), 2);

    EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
    EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);
    EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);

    EXPECT_EQ(result.metadata.schema_info[0].name, "col1");
    EXPECT_EQ(result.metadata.schema_info[1].name, "col2");
    EXPECT_EQ(result.metadata.schema_info[2].name, "col3");

    auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}, validity});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                   float64_wrapper{{1.1, 2.2}, validity});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                   cudf::test::strings_column_wrapper({"aaa", "bbb"}));
  };
  // simple case
  test_json_objects(
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col1\":200, \"col2\":2.2, \"col3\":\"bbb\"}\n");
  // out of order fields
  test_json_objects(
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col3\":\"bbb\", \"col1\":200, \"col2\":2.2}\n");
}

TEST_F(JsonReaderTest, JsonLinesObjectsMissingData)
{
  // Note: columns will be ordered based on which fields appear first
  std::string const data =
    "{              \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col1\":200,               \"col3\":\"bbb\"}\n";
  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
      .lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col2");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col3");
  EXPECT_EQ(result.metadata.schema_info[2].name, "col1");

  auto col1_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 0; });
  auto col2_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i == 0; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 float64_wrapper{{0., 200.}, col1_validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 float64_wrapper{{1.1, 0.}, col2_validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 cudf::test::strings_column_wrapper({"aaa", "bbb"}));
}

TEST_F(JsonReaderTest, JsonLinesObjectsOutOfOrder)
{
  std::string const data =
    "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
    "{\"col3\":\"bbb\", \"col1\":200, \"col2\":2.2}\n";

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{data.data(), data.size()})
      .lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "col1");
  EXPECT_EQ(result.metadata.schema_info[1].name, "col2");
  EXPECT_EQ(result.metadata.schema_info[2].name, "col3");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int64_wrapper{{100, 200}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(2),
                                 cudf::test::strings_column_wrapper({"aaa", "bbb"}));
}

/*
// currently, the json reader is strict about having non-empty input.
TEST_F(JsonReaderTest, EmptyFile)
{
  auto filepath = temp_env->get_temp_dir() + "EmptyFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath}).lines(true);
  auto result = cudf_io::read_json(in_options);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}

// currently, the json reader is strict about having non-empty input.
TEST_F(JsonReaderTest, NoDataFile)
{
  auto filepath = temp_env->get_temp_dir() + "NoDataFile.csv";
  {
    std::ofstream outfile{filepath, std::ofstream::out};
    outfile << "{}\n";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath}).lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  const auto view = result.tbl->view();
  EXPECT_EQ(0, view.num_columns());
}
*/

TEST_F(JsonReaderTest, ArrowFileSource)
{
  const std::string fname = temp_env->get_temp_dir() + "ArrowFileSource.csv";

  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[9]\n[8]\n[7]\n[6]\n[5]\n[4]\n[3]\n[2]\n";
  outfile.close();

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(fname).Value(&infile).ok());

  auto arrow_source = cudf_io::arrow_io_source{infile};
  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{&arrow_source})
      .dtypes({dtype<int8_t>()})
      .lines(true);
  ;
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT8);

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int8_wrapper{{9, 8, 7, 6, 5, 4, 3, 2}, validity});
}

TEST_F(JsonReaderTest, InvalidFloatingPoint)
{
  const auto filepath = temp_env->get_temp_dir() + "InvalidFloatingPoint.json";
  {
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << "[1.2e1+]\n[3.4e2-]\n[5.6e3e]\n[7.8e3A]\n[9.0Be1]\n[1C.2]";
  }

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath})
      .dtypes({dtype<float>()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::FLOAT32);

  const auto col_data = cudf::test::to_host<float>(result.tbl->view().column(0));
  // col_data.first contains the column data
  for (const auto& elem : col_data.first)
    ASSERT_TRUE(std::isnan(elem));
  // col_data.second contains the bitmasks
  ASSERT_EQ(0u, col_data.second[0]);
}

TEST_F(JsonReaderTest, StringInference)
{
  std::string buffer = "[\"-1\"]";
  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .lines(true);
  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
}

TEST_F(JsonReaderTest, ParseInRangeIntegers)
{
  constexpr auto num_rows                      = 4;
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

  auto filepath = temp_env->get_temp_dir() + "ParseInRangeIntegers.json";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << "[" << small_int[i] << "," << less_equal_int64_max[i] << ","
           << greater_equal_int64_min[i] << "," << greater_int64_max[i] << ","
           << less_equal_uint64_max[i] << "," << small_int_append_zeros[i] << ","
           << less_equal_int64_max_append_zeros[i] << "," << greater_equal_int64_min_append_zeros[i]
           << "," << greater_int64_max_append_zeros[i] << ","
           << less_equal_uint64_max_append_zeros[i] << "]\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }
  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath}).lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

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

TEST_F(JsonReaderTest, ParseOutOfRangeIntegers)
{
  constexpr auto num_rows                        = 4;
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

  auto filepath = temp_env->get_temp_dir() + "ParseOutOfRangeIntegers.json";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << "[" << out_of_range_positive[i] << "," << out_of_range_negative[i] << ","
           << greater_uint64_max[i] << "," << less_int64_min[i] << "," << mixed_range[i] << ","
           << out_of_range_positive_append_zeros[i] << "," << out_of_range_negative_append_zeros[i]
           << "," << greater_uint64_max_append_zeros[i] << "," << less_int64_min_append_zeros[i]
           << "," << mixed_range_append_zeros[i] << "]\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }
  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{filepath}).lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

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

TEST_F(JsonReaderTest, JsonLinesMultipleFileInputs)
{
  const std::string file1 = temp_env->get_temp_dir() + "JsonLinesFileTest1.json";
  std::ofstream outfile(file1, std::ofstream::out);
  outfile << "[11, 1.1]\n[22, 2.2]\n";
  outfile.close();

  const std::string file2 = temp_env->get_temp_dir() + "JsonLinesFileTest2.json";
  std::ofstream outfile2(file2, std::ofstream::out);
  outfile2 << "[33, 3.3]\n[44, 4.4]";
  outfile2.close();

  cudf_io::json_reader_options in_options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{{file1, file2}}).lines(true);

  cudf_io::table_with_metadata result = cudf_io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 4);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 int64_wrapper{{11, 22, 33, 44}, validity});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 float64_wrapper{{1.1, 2.2, 3.3, 4.4}, validity});
}

TEST_F(JsonReaderTest, BadDtypeParams)
{
  std::string buffer = "[1,2,3,4]";

  cudf_io::json_reader_options options_vec =
    cudf_io::json_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .lines(true)
      .dtypes({dtype<int8_t>()});

  // should throw because there are four columns and only one dtype
  EXPECT_THROW(cudf_io::read_json(options_vec), cudf::logic_error);

  cudf_io::json_reader_options options_map =
    cudf_io::json_reader_options::builder(cudf_io::source_info{buffer.c_str(), buffer.size()})
      .lines(true)
      .dtypes(std::map<std::string, cudf::data_type>{{"0", dtype<int8_t>()},
                                                     {"1", dtype<int8_t>()},
                                                     {"2", dtype<int8_t>()},
                                                     {"wrong_name", dtype<int8_t>()}});
  // should throw because one of the columns is not in the dtype map
  EXPECT_THROW(cudf_io::read_json(options_map), cudf::logic_error);
}

TEST_F(JsonReaderTest, JsonExperimentalBasic)
{
  std::string const fname = temp_env->get_temp_dir() + "JsonExperimentalBasic.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << R"([{"a":"11", "b":"1.1"},{"a":"22", "b":"2.2"}])";
  outfile.close();

  cudf_io::json_reader_options options =
    cudf_io::json_reader_options::builder(cudf_io::source_info{fname}).experimental(true);
  auto result = cudf_io::read_json(options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::STRING);

  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "b");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 cudf::test::strings_column_wrapper({"11", "22"}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1),
                                 cudf::test::strings_column_wrapper({"1.1", "2.2"}));
}

TEST_F(JsonReaderTest, JsonExperimentalLines)
{
  std::string const json_string =
    R"({"a":"a0"}
    {"a":"a1"}
    {"a":"a2", "b":"b2"}
    {"a":"a3", "c":"c3"}
    {"a":"a4"})";

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{json_string.c_str(), json_string.size()})
      .lines(true);

  // Read test data via existing, non-nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

  // Read test data via new, nested JSON reader
  json_lines_options.enable_experimental(true);
  cudf::io::table_with_metadata new_reader_table = cudf::io::read_json(json_lines_options);

  // Verify that the data read via non-nested JSON lines reader matches the data read via nested
  // JSON reader
  CUDF_TEST_EXPECT_TABLES_EQUAL(current_reader_table.tbl->view(), new_reader_table.tbl->view());
}

TEST_F(JsonReaderTest, ExperimentalLinesNoOmissions)
{
  std::array<std::string const, 4> const json_inputs
    // single column
    {R"({"a":"a0"}
    {"a":"a1"}
    {"a":"a2"}
    {"a":"a3"}
    {"a":"a4"})",
     // single column, single row
     R"({"a":"a0"})",
     // single row
     R"({"a":"a0", "b":"b0"})",
     // two column, two rows
     R"({"a":"a0", "b":"b0"}
    {"a":"a1", "b":"b1"})"};

  for (auto const& json_string : json_inputs) {
    // Initialize parsing options (reading json lines)
    cudf::io::json_reader_options json_lines_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{json_string.c_str(), json_string.size()})
        .lines(true);

    // Read test data via existing, non-nested JSON lines reader
    cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

    // Read test data via new, nested JSON reader
    json_lines_options.enable_experimental(true);
    cudf::io::table_with_metadata new_reader_table = cudf::io::read_json(json_lines_options);

    // Verify that the data read via non-nested JSON lines reader matches the data read via nested
    // JSON reader
    CUDF_TEST_EXPECT_TABLES_EQUAL(current_reader_table.tbl->view(), new_reader_table.tbl->view());
  }
}

TEST_F(JsonReaderTest, TestColumnOrder)
{
  std::string const json_string =
    // Expected order:
    // root: b, c, a, d
    // a: 2, 0, 1
    {R"({"b":"b0"}
    {"c":"c1","a":{"2":null}}
    {"d":"d2","a":{"0":"a2.0", "2":"a2.2"}}
    {"b":"b3","a":{"1":null, "2":"a3.2"}})"};

  std::vector<std::string> const root_col_names{"b", "c", "a", "d"};
  std::vector<std::string> const a_child_col_names{"2", "0", "1"};

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{json_string.c_str(), json_string.size()})
      .lines(true)
      .experimental(true);

  // Read in data using nested JSON reader
  cudf::io::table_with_metadata new_reader_table = cudf::io::read_json(json_lines_options);

  // Verify root column order (assert to avoid OOB access)
  ASSERT_EQ(new_reader_table.metadata.schema_info.size(), root_col_names.size());

  for (std::size_t i = 0; i < a_child_col_names.size(); i++) {
    auto const& root_col_name = root_col_names[i];
    EXPECT_EQ(new_reader_table.metadata.schema_info[i].name, root_col_name);
  }

  // Verify nested child column order (assert to avoid OOB access)
  ASSERT_EQ(new_reader_table.metadata.schema_info[2].children.size(), a_child_col_names.size());
  for (std::size_t i = 0; i < a_child_col_names.size(); i++) {
    auto const& a_child_col_name = a_child_col_names[i];
    EXPECT_EQ(new_reader_table.metadata.schema_info[2].children[i].name, a_child_col_name);
  }

  // Verify data of root columns
  ASSERT_EQ(root_col_names.size(), new_reader_table.tbl->num_columns());
  column_wrapper<cudf::string_view> root_col_data_b{{"b0", "", "", "b3"},
                                                    {true, false, false, true}};
  column_wrapper<cudf::string_view> root_col_data_c{{"", "c1", "", ""},
                                                    {false, true, false, false}};
  column_wrapper<cudf::string_view> root_col_data_d{{"", "", "d2", ""},
                                                    {false, false, true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_b, new_reader_table.tbl->get_column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_c, new_reader_table.tbl->get_column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(root_col_data_d, new_reader_table.tbl->get_column(3));

  // Verify data of child columns of column 'a'
  auto const col_a = new_reader_table.tbl->get_column(2);
  ASSERT_EQ(a_child_col_names.size(), col_a.num_children());
  column_wrapper<cudf::string_view> col_a2{{"", "", "a2.2", "a3.2"}, {false, false, true, true}};
  column_wrapper<cudf::string_view> col_a0{{"", "", "a2.0", ""}, {false, false, true, false}};
  // col a.1 is inferred as all-null
  int8_wrapper col_a1{{0, 0, 0, 0}, {false, false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a2, col_a.child(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a0, col_a.child(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_a1, col_a.child(2));
}

CUDF_TEST_PROGRAM_MAIN()
