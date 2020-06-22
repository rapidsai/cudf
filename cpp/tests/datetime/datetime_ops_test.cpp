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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/timestamp_utilities.cuh>
#include <tests/utilities/type_lists.hpp>

template <typename T>
struct NonTimestampTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using NonTimestampTypes = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::StringTypes>;

TYPED_TEST_CASE(NonTimestampTest, NonTimestampTypes);

TYPED_TEST(NonTimestampTest, TestThrowsOnNonTimestamp)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  cudf::data_type dtype{cudf::type_to_id<T>()};
  cudf::column col{dtype, 0, rmm::device_buffer{0}};

  EXPECT_THROW(extract_year(col), cudf::logic_error);
  EXPECT_THROW(extract_month(col), cudf::logic_error);
  EXPECT_THROW(extract_day(col), cudf::logic_error);
  EXPECT_THROW(extract_weekday(col), cudf::logic_error);
  EXPECT_THROW(extract_hour(col), cudf::logic_error);
  EXPECT_THROW(extract_minute(col), cudf::logic_error);
  EXPECT_THROW(extract_second(col), cudf::logic_error);
  EXPECT_THROW(last_day_of_month(col), cudf::logic_error);
  EXPECT_THROW(day_of_year(col), cudf::logic_error);
}

struct BasicDatetimeOpsTest : public cudf::test::BaseFixture {
};

TEST_F(BasicDatetimeOpsTest, TestExtractingDatetimeComponents)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  auto timestamps_D = fixed_width_column_wrapper<cudf::timestamp_D>{
    -1528,  // 1965-10-26 GMT
    17716,  // 2018-07-04 GMT
    19382,  // 2023-01-25 GMT
  };

  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s>{
    -131968728,  // 1965-10-26 14:01:12 GMT
    1530705600,  // 2018-07-04 12:00:00 GMT
    1674631932,  // 2023-01-25 07:32:12 GMT
  };

  auto timestamps_ms = fixed_width_column_wrapper<cudf::timestamp_ms>{
    -131968727238,  // 1965-10-26 14:01:12.762 GMT
    1530705600000,  // 2018-07-04 12:00:00.000 GMT
    1674631932929,  // 2023-01-25 07:32:12.929 GMT
  };

  expect_columns_equal(*extract_year(timestamps_D),
                       fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  expect_columns_equal(*extract_year(timestamps_s),
                       fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  expect_columns_equal(*extract_year(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});

  expect_columns_equal(*extract_month(timestamps_D), fixed_width_column_wrapper<int16_t>{10, 7, 1});
  expect_columns_equal(*extract_month(timestamps_s), fixed_width_column_wrapper<int16_t>{10, 7, 1});
  expect_columns_equal(*extract_month(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{10, 7, 1});

  expect_columns_equal(*extract_day(timestamps_D), fixed_width_column_wrapper<int16_t>{26, 4, 25});
  expect_columns_equal(*extract_day(timestamps_s), fixed_width_column_wrapper<int16_t>{26, 4, 25});
  expect_columns_equal(*extract_day(timestamps_ms), fixed_width_column_wrapper<int16_t>{26, 4, 25});

  expect_columns_equal(*extract_weekday(timestamps_D),
                       fixed_width_column_wrapper<int16_t>{2, 3, 3});
  expect_columns_equal(*extract_weekday(timestamps_s),
                       fixed_width_column_wrapper<int16_t>{2, 3, 3});
  expect_columns_equal(*extract_weekday(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{2, 3, 3});

  expect_columns_equal(*extract_hour(timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*extract_hour(timestamps_s), fixed_width_column_wrapper<int16_t>{14, 12, 7});
  expect_columns_equal(*extract_hour(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{14, 12, 7});

  expect_columns_equal(*extract_minute(timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*extract_minute(timestamps_s),
                       fixed_width_column_wrapper<int16_t>{1, 0, 32});
  expect_columns_equal(*extract_minute(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{1, 0, 32});

  expect_columns_equal(*extract_second(timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*extract_second(timestamps_s),
                       fixed_width_column_wrapper<int16_t>{12, 0, 12});
  expect_columns_equal(*extract_second(timestamps_ms),
                       fixed_width_column_wrapper<int16_t>{12, 0, 12});
}

template <typename T>
struct TypedDatetimeOpsTest : public cudf::test::BaseFixture {
  cudaStream_t stream() { return cudaStream_t(0); }
  cudf::size_type size() { return cudf::size_type(10); }
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_CASE(TypedDatetimeOpsTest, cudf::test::TimestampTypes);

TYPED_TEST(TypedDatetimeOpsTest, TestEmptyColumns)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  auto int16s_dtype     = cudf::data_type{cudf::type_to_id<int16_t>()};
  auto timestamps_dtype = cudf::data_type{cudf::type_to_id<T>()};

  cudf::column int16s{int16s_dtype, 0, rmm::device_buffer{0}};
  cudf::column timestamps{timestamps_dtype, 0, rmm::device_buffer{0}};

  expect_columns_equal(*extract_year(timestamps), int16s);
  expect_columns_equal(*extract_month(timestamps), int16s);
  expect_columns_equal(*extract_day(timestamps), int16s);
  expect_columns_equal(*extract_weekday(timestamps), int16s);
  expect_columns_equal(*extract_hour(timestamps), int16s);
  expect_columns_equal(*extract_minute(timestamps), int16s);
  expect_columns_equal(*extract_second(timestamps), int16s);
}

TYPED_TEST(TypedDatetimeOpsTest, TestExtractingGeneratedDatetimeComponents)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamps =
    generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop_));

  auto expected_years =
    fixed_width_column_wrapper<int16_t>{1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033};
  auto expected_months   = fixed_width_column_wrapper<int16_t>{10, 8, 6, 4, 2, 1, 11, 9, 7, 5};
  auto expected_days     = fixed_width_column_wrapper<int16_t>{11, 16, 20, 24, 26, 1, 5, 9, 14, 18};
  auto expected_weekdays = fixed_width_column_wrapper<int16_t>{6, 4, 2, 7, 5, 4, 2, 7, 5, 3};
  auto expected_hours    = fixed_width_column_wrapper<int16_t>{19, 20, 21, 22, 23, 0, 0, 1, 2, 3};
  auto expected_minutes = fixed_width_column_wrapper<int16_t>{33, 26, 20, 13, 6, 0, 53, 46, 40, 33};
  auto expected_seconds = fixed_width_column_wrapper<int16_t>{20, 40, 0, 20, 40, 0, 20, 40, 0, 20};

  // Special cases for timestamp_D: zero out the expected hh/mm/ss cols
  if (std::is_same<TypeParam, cudf::timestamp_D>::value) {
    expected_hours   = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_minutes = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_seconds = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  }

  expect_columns_equal(*extract_year(timestamps), expected_years);
  expect_columns_equal(*extract_month(timestamps), expected_months);
  expect_columns_equal(*extract_day(timestamps), expected_days);
  expect_columns_equal(*extract_weekday(timestamps), expected_weekdays);
  expect_columns_equal(*extract_hour(timestamps), expected_hours);
  expect_columns_equal(*extract_minute(timestamps), expected_minutes);
  expect_columns_equal(*extract_second(timestamps), expected_seconds);
}

TYPED_TEST(TypedDatetimeOpsTest, TestExtractingGeneratedNullableDatetimeComponents)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamps =
    generate_timestamps<T, true>(this->size(), time_point_ms(start), time_point_ms(stop_));

  auto expected_years = fixed_width_column_wrapper<int16_t>{
    {1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_months = fixed_width_column_wrapper<int16_t>{
    {10, 8, 6, 4, 2, 1, 11, 9, 7, 5},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_days = fixed_width_column_wrapper<int16_t>{
    {11, 16, 20, 24, 26, 1, 5, 9, 14, 18},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_weekdays = fixed_width_column_wrapper<int16_t>{
    {6, 4, 2, 7, 5, 4, 2, 7, 5, 3},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_hours = fixed_width_column_wrapper<int16_t>{
    {19, 20, 21, 22, 23, 0, 0, 1, 2, 3},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_minutes = fixed_width_column_wrapper<int16_t>{
    {33, 26, 20, 13, 6, 0, 53, 46, 40, 33},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_seconds = fixed_width_column_wrapper<int16_t>{
    {20, 40, 0, 20, 40, 0, 20, 40, 0, 20},
    {true, false, true, false, true, false, true, false, true, false}};

  // Special cases for timestamp_D: zero out the expected hh/mm/ss cols
  if (std::is_same<TypeParam, cudf::timestamp_D>::value) {
    expected_hours = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
    expected_minutes = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
    expected_seconds = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
  }

  expect_columns_equal(*extract_year(timestamps), expected_years);
  expect_columns_equal(*extract_month(timestamps), expected_months);
  expect_columns_equal(*extract_day(timestamps), expected_days);
  expect_columns_equal(*extract_weekday(timestamps), expected_weekdays);
  expect_columns_equal(*extract_hour(timestamps), expected_hours);
  expect_columns_equal(*extract_minute(timestamps), expected_minutes);
  expect_columns_equal(*extract_second(timestamps), expected_seconds);
}

TEST_F(BasicDatetimeOpsTest, TestLastDayOfMonthWithSeconds)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s>{
    662688000L,   // 1991-01-01 00:00:00 GMT
    949496401L,   // 2000-02-02 13:00:01 GMT - leap year
    4106854801L,  // 2100-02-21 01:00:01 GMT - not a leap year
    1582391837L,  // 2020-02-22 17:17:17 GMT - leap year
    1363046401L,  // 2013-03-12 00:00:01 GMT
    1302696000L,  // 2011-04-13 12:00:00 GMT
    1495800001L,  // 2017-05-26 12:00:01 GMT
    1056931201L,  // 2003-06-30 00:00:01 GMT - already last day
    1031961599L,  // 2002-09-13 23:59:59 GMT
    0L,           // This is the UNIX epoch - 1970-01-01
    -131968728L,  // 1965-10-26 14:01:12 GMT
  };

  expect_columns_equal(*last_day_of_month(timestamps_s),
                       fixed_width_column_wrapper<cudf::timestamp_D>{
                         7700,   // 1991-01-31
                         11016,  // 2000-02-29
                         47540,  // 2100-02-28
                         18321,  // 2020-02-29
                         15795,  // 2013-03-31
                         15094,  // 2011-04-30
                         17317,  // 2017-05-31
                         12233,  // 2003-06-30
                         11960,  // 2002-09-30
                         30,     // This is the UNIX epoch - when rounded up becomes 1970-01-31
                         -1523   // 1965-10-31
                       },
                       true);
}

TEST_F(BasicDatetimeOpsTest, TestLastDayOfMonthWithDate)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  // Time in days since epoch
  // Dates converted using epochconverter.com
  // Make some nullable fields as well
  auto timestamps_d = fixed_width_column_wrapper<cudf::timestamp_D>{
    {
      999,    // Random nullable field
      0,      // This is the UNIX epoch - 1970-01-01
      44376,  // 2091-07-01 00:00:00 GMT
      47695,  // 2100-08-02 00:00:00 GMT
      3,      // Random nullable field
      66068,  // 2150-11-21 00:00:00 GMT
      22270,  // 2030-12-22 00:00:00 GMT
      111,    // Random nullable field
    },
    {false, true, true, true, false, true, true, false},
  };

  expect_columns_equal(*last_day_of_month(timestamps_d),
                       fixed_width_column_wrapper<cudf::timestamp_D>{
                         {
                           999,    // Random nullable field
                           30,     // This is the UNIX epoch - when rounded up becomes 1970-01-31
                           44406,  // 2091-07-31
                           47724,  // 2100-08-31
                           3,      // Random nullable field
                           66077,  // 2150-11-30
                           22279,  // 2030-12-31
                           111,    // Random nullable field
                         },
                         {false, true, true, true, false, true, true, false},
                       },
                       true);
}

TEST_F(BasicDatetimeOpsTest, TestDayOfYearWithDate)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  // Day number in the year
  // Dates converted using epochconverter.com
  // Make some nullable fields as well
  auto timestamps_d = fixed_width_column_wrapper<cudf::timestamp_s>{
    {
      999L,         // Random nullable field
      0L,           // This is the UNIX epoch - 1970-01-01
      1577865600L,  // 2020-01-01 00:00:00 GMT
      1581667200L,  // 2020-02-14 00:00:00 GMT
      3L,           // Random nullable field
      1609401600L,  // 2020-12-31 00:00:00 GMT
      4133923200L,  // 2100-12-31 00:00:00 GMT
      111L,         // Random nullable field
      -2180188800L  // 1900-11-30 00:00:00 GMT
    },
    {false, true, true, true, false, true, true, false, true},
  };

  expect_columns_equal(*day_of_year(timestamps_d),
                       fixed_width_column_wrapper<int16_t>{
                         {
                           999,  // Random nullable field
                           1,    // Number of year days until UNIX epoch time
                           1,    // Number of year days until 2020-01-01
                           45,   // Number of year days until 2020-02-14
                           3,    // Random nullable field
                           366,  // Number of year days until 2020-12-31
                           365,  // Number of year days until 2100-12-31
                           111,  // Random nullable field
                           334   // Number of year days until 1900-11-30
                         },
                         {false, true, true, true, false, true, true, false, true},
                       },
                       true);
}

TEST_F(BasicDatetimeOpsTest, TestDayOfYearWithEmptyColumn)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace simt::std::chrono;

  // Create an empty column
  auto timestamps_d = fixed_width_column_wrapper<cudf::timestamp_s>{};
  auto out_col      = day_of_year(timestamps_d);
  EXPECT_EQ(out_col->size(), 0);
}

CUDF_TEST_PROGRAM_MAIN()
