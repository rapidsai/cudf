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

#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct StringsDatetimeTest : public cudf::test::BaseFixture {
};

TEST_F(StringsDatetimeTest, ToTimestamp)
{
  std::vector<const char*> h_strings{"1974-02-28T01:23:45Z",
                                     "2019-07-17T21:34:37Z",
                                     nullptr,
                                     "",
                                     "2019-03-20T12:34:56Z",
                                     "2020-02-29T00:00:00Z",
                                     "1921-01-07T14:32:07Z",
                                     "1969-12-31T23:59:45Z"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<cudf::timestamp_s> h_expected{
    131246625, 1563399277, 0, 0, 1553085296, 1582934400, -1545730073, -15};

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%Y-%m-%dT%H:%M:%SZ");

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, ToTimestampAmPm)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45 PM",
                                             "2019-07-17 02:34:56 AM",
                                             "2019-03-20 12:34:56 PM",
                                             "2020-02-29 12:00:00 AM",
                                             "1925-02-07 02:55:08 PM"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%Y-%m-%d %I:%M:%S %p");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> expected{
    131289825, 1563330896, 1553085296, 1582934400, -1416819892};
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, ToTimestampMicrosecond)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45.987000",
                                             "2019-07-17 02:34:56.001234",
                                             "2019-03-20 12:34:56.100100",
                                             "2020-02-29 00:00:00.555777",
                                             "1969-12-31 00:00:01.000055",
                                             "1944-07-21 11:15:09.333444"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_MILLISECONDS}, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms> expected_ms{
    131246625987, 1563330896001, 1553085296100, 1582934400555, -86399000L, -803047490667L};
  cudf::test::expect_columns_equal(*results, expected_ms);
  results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS}, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns> expected_ns{131246625987000000,
                                                                         1563330896001234000,
                                                                         1553085296100100000,
                                                                         1582934400555777000,
                                                                         -86398999945000,
                                                                         -803047490666556000};
  cudf::test::expect_columns_equal(*results, expected_ns);
}

TEST_F(StringsDatetimeTest, ToTimestampMillisecond)
{
  cudf::test::strings_column_wrapper strings{"2018-07-04 12:00:00.123",
                                             "2020-04-06 13:09:00.555",
                                             "1969-12-31 00:00:00.000",
                                             "1956-01-23 17:18:19.000"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_MICROSECONDS}, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us> expected_us{
    1530705600123000, 1586178540555000, -86400000000, -439886501000000};
  cudf::test::expect_columns_equal(*results, expected_us);
  results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS}, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns> expected_ns{
    1530705600123000000, 1586178540555000000, -86400000000000, -439886501000000000};
  cudf::test::expect_columns_equal(*results, expected_ns);
}

TEST_F(StringsDatetimeTest, ToTimestampTimezone)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45+0100",
                                             "2019-07-17 02:34:56-0300",
                                             "2019-03-20 12:34:56+1030",
                                             "2020-02-29 12:00:00-0500",
                                             "1938-11-23 10:28:49+0700"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%Y-%m-%d %H:%M:%S%z");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> expected{
    131243025, 1563341696, 1553047496, 1582995600, -981664271};
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestamp)
{
  std::vector<cudf::timestamp_s> h_timestamps{
    131246625, 1563399277, 0, 1553085296, 1582934400, -1545730073, -86399};
  std::vector<const char*> h_expected{"1974-02-28T01:23:45Z",
                                      "2019-07-17T21:34:37Z",
                                      nullptr,
                                      "2019-03-20T12:34:56Z",
                                      "2020-02-29T00:00:00Z",
                                      "1921-01-07T14:32:07Z",
                                      "1969-12-31T00:00:01Z"};

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps(
    h_timestamps.begin(),
    h_timestamps.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_timestamps(timestamps);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampAmPm)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps{
    1530705600L, 1582934461L, 1451430122L, 1318302183L, -6105994200L};
  auto results = cudf::strings::from_timestamps(timestamps, "%Y-%m-%d %I:%M:%S %p");
  cudf::test::strings_column_wrapper expected{"2018-07-04 12:00:00 PM",
                                              "2020-02-29 12:01:01 AM",
                                              "2015-12-29 11:02:02 PM",
                                              "2011-10-11 03:03:03 AM",
                                              "1776-07-04 06:30:00 PM"};
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampMillisecond)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms> timestamps_ms{
    1530705600123, 1582934461007, 1451430122421, 1318302183999, -6106017600047};
  auto results = cudf::strings::from_timestamps(timestamps_ms, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::strings_column_wrapper expected_ms{"2018-07-04 12:00:00.123",
                                                 "2020-02-29 00:01:01.007",
                                                 "2015-12-29 23:02:02.421",
                                                 "2011-10-11 03:03:03.999",
                                                 "1776-07-04 12:00:00.953"};
  cudf::test::expect_columns_equal(*results, expected_ms);

  results = cudf::strings::from_timestamps(timestamps_ms, "%Y-%m-%d %H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ms_6f{"2018-07-04 12:00:00.123000",
                                                    "2020-02-29 00:01:01.007000",
                                                    "2015-12-29 23:02:02.421000",
                                                    "2011-10-11 03:03:03.999000",
                                                    "1776-07-04 12:00:00.953000"};
  cudf::test::expect_columns_equal(*results, expected_ms_6f);

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns> timestamps_ns{1530705600123456789,
                                                                           1582934461007008009,
                                                                           1451430122421310209,
                                                                           1318302183999777555,
                                                                           -6106017600047047047};
  results = cudf::strings::from_timestamps(timestamps_ns, "%Y-%m-%d %H:%M:%S.%9f");
  cudf::test::strings_column_wrapper expected_ns{"2018-07-04 12:00:00.123456789",
                                                 "2020-02-29 00:01:01.007008009",
                                                 "2015-12-29 23:02:02.421310209",
                                                 "2011-10-11 03:03:03.999777555",
                                                 "1776-07-04 12:00:00.952952953"};
  cudf::test::expect_columns_equal(*results, expected_ns);

  results = cudf::strings::from_timestamps(timestamps_ns, "%Y-%m-%d %H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ns_6f{"2018-07-04 12:00:00.123456",
                                                    "2020-02-29 00:01:01.007008",
                                                    "2015-12-29 23:02:02.421310",
                                                    "2011-10-11 03:03:03.999777",
                                                    "1776-07-04 12:00:00.952952"};
  cudf::test::expect_columns_equal(*results, expected_ns_6f);
}

TEST_F(StringsDatetimeTest, FromTimestampTimezone)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps{
    1530705600L, 1582934461L, 1451430122L, 1318302183L, -2658802500L};
  auto results = cudf::strings::from_timestamps(timestamps, "%m/%d/%y %H%M%S%z");
  cudf::test::strings_column_wrapper expected{"07/04/18 120000+0000",
                                              "02/29/20 000101+0000",
                                              "12/29/15 230202+0000",
                                              "10/11/11 030303+0000",
                                              "09/29/85 194500+0000"};
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampDayOfYear)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps{
    118800L,       // 1970-01-02 09:00:00
    1293901860L,   // 2011-01-01 17:11:00
    318402000L,    // 1980-02-03 05:00:00
    604996200L,    // 1989-03-04 06:30:00
    1270413572L,   // 2010-04-04 20:39:32
    1588734621L,   // 2020-05-06 03:10:21
    2550814152L,   // 2050-10-31 07:29:12
    4102518778L,   // 2100-01-01 20:32:58
    702696234L,    // 1992-04-08 01:23:54
    6516816203L,   // 2176-07-05 02:43:23
    26472091292L,  // 2808-11-12 22:41:32
    4133857172L,   // 2100-12-30 01:39:32
    1560948892L,   // 2019-06-19 12:54:52
    4115217600L,   // 2100-05-28 20:00:00
    -265880250L,   // 1961-07-29 16:22:30
  };
  auto results = cudf::strings::from_timestamps(timestamps, "%d/%m/%Y %j");
  cudf::test::strings_column_wrapper expected{"02/01/1970 002",
                                              "01/01/2011 001",
                                              "03/02/1980 034",
                                              "04/03/1989 063",
                                              "04/04/2010 094",
                                              "06/05/2020 127",
                                              "31/10/2050 304",
                                              "01/01/2100 001",
                                              "08/04/1992 099",
                                              "05/07/2176 187",
                                              "12/11/2808 317",
                                              "30/12/2100 364",
                                              "19/06/2019 170",
                                              "28/05/2100 148",
                                              "29/07/1961 210"};
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsDatetimeTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::TIMESTAMP_SECONDS}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::from_timestamps(zero_size_column);
  cudf::test::expect_strings_empty(results->view());

  cudf::column_view zero_size_strings_column(cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
  results = cudf::strings::to_timestamps(cudf::strings_column_view(zero_size_strings_column),
                                         cudf::data_type{cudf::TIMESTAMP_SECONDS},
                                         "%Y");
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsDatetimeTest, Errors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view view(strings);
  EXPECT_THROW(cudf::strings::to_timestamps(view, cudf::data_type{cudf::INT64}, "%Y"),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_timestamps(view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, ""),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_timestamps(view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%2Y"),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_timestamps(view, cudf::data_type{cudf::TIMESTAMP_SECONDS}, "%g"),
               cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int64_t> invalid_timestamps{1530705600};
  EXPECT_THROW(cudf::strings::from_timestamps(invalid_timestamps), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s> timestamps{1530705600};
  EXPECT_THROW(cudf::strings::from_timestamps(timestamps, ""), cudf::logic_error);
}
