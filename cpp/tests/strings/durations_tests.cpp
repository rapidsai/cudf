/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/durations.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

#define CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, rhs) \
  {                                              \
    SCOPED_TRACE(" <--  line of failure\n");     \
    cudf::test::expect_columns_equal(lhs, rhs);  \
  }

struct StringsDurationsTest : public cudf::test::BaseFixture {
};

TEST_F(StringsDurationsTest, FromDurations)
{
  using T = cudf::duration_s;
  std::vector<cudf::duration_s> h_durations{
    T{131246625}, T{1563399277}, T{0}, T{1553085296}, T{1582934400}, T{-1545730073}, T{-86399}};
  std::vector<const char*> h_expected{"1519 days 01:23:45",
                                      "18094 days 21:34:37",
                                      nullptr,
                                      "17975 days 12:34:56",
                                      "18321 days 00:00:00",
                                      "-17891 days +14:32:07",
                                      "-1 days +00:00:01"};

  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations(
    h_durations.begin(),
    h_durations.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_durations(durations);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDurationsTest, ISOFormat)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{1530705600L}, T{1582934461L}, T{1451430122L}, T{1318302183L}, T{-6105994200L}};
  auto results = cudf::strings::from_durations(durations, "P%dDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected{
    "P17716DT12H0M0S", "P18321DT0H1M1S", "P16798DT23H2M2S", "P15258DT3H3M3S", "P-70672DT18H30M0S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDurationsTest, ISOFormatDaysOnly)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{17716L}, T{18321L}, T{16798L}, T{15258L}, T{-70672L}};
  auto results1 = cudf::strings::from_durations(durations, "P%dDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected1{
    "P17716DT0H0M0S", "P18321DT0H0M0S", "P16798DT0H0M0S", "P15258DT0H0M0S", "P-70672DT0H0M0S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results1, expected1);
  auto results2 = cudf::strings::from_durations(durations, "P%dD");
  cudf::test::strings_column_wrapper expected2{
    "P17716D", "P18321D", "P16798D", "P15258D", "P-70672D"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results2, expected2);
}

TEST_F(StringsDurationsTest, ISOFormatSubseconds)
{
  using T = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},  T{7000000000L}, T{11L}, T{10L}, T{17716L}, T{18321L}, T{16798L}, T{15258L}, T{15258000L}, T{-70672L}};
  auto results = cudf::strings::from_durations(durations, "P%dDT%HH%MM%S.%fS");
  cudf::test::strings_column_wrapper expected{"P0DT0H0M0S",
                                              "P0DT0H0M7S",
                                              "P0DT0H0M0.000000011S",
                                              "P0DT0H0M0.00000001S",
                                              "P0DT0H0M0.000017716S",
                                              "P0DT0H0M0.000018321S",
                                              "P0DT0H0M0.000016798S",
                                              "P0DT0H0M0.000015258S",
                                              "P0DT0H0M0.015258S",
                                              "P-1DT23H59M59.999929328S"};
  cudf::test::print(*results);
  cudf::test::print(expected);
  //CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto results2 = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S.%u");
  cudf::test::strings_column_wrapper expected2{"00:00:00",
                                               "00:00:07",
                                               "00:00:00.000000",
                                               "00:00:00.000000",
                                               "00:00:00.000017",
                                               "00:00:00.000018",
                                               "00:00:00.000016",
                                               "00:00:00.000015",
                                               "00:00:00.015258",
                                               "-1 days +23:59:59.999929"};
  cudf::test::print(*results2);
  cudf::test::print(expected2);
}


TEST_F(StringsDurationsTest, FromDurationSubseconds)
{
  using ms = cudf::duration_ms;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> durations_ms{ms{-60000},
                                                                         ms{1530705600123},
                                                                         ms{1582934461007},
                                                                         ms{1451430122420},
                                                                         ms{1318302183999},
                                                                         ms{-6106017600047}};
  auto results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S.%3f");
  cudf::test::strings_column_wrapper expected_ms_3f{"-1 days +23:59:00.000",
                                                 "17716 days 12:00:00.123",
                                                 "18321 days 00:01:01.007",
                                                 "16798 days 23:02:02.420",
                                                 "15258 days 03:03:03.999",
                                                 "-70672 days +11:59:59.953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms_3f);

  results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S.%6f");
  cudf::test::strings_column_wrapper expected_ms_6f{"-1 days +23:59:00.000000",
                                                    "17716 days 12:00:00.123000",
                                                    "18321 days 00:01:01.007000",
                                                    "16798 days 23:02:02.420000",
                                                    "15258 days 03:03:03.999000",
                                                    "-70672 days +11:59:59.953000"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms_6f);

    results = cudf::strings::from_durations(durations_ms, "%d days %+%H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ms{"-1 days +23:59:00.",
                                                    "17716 days 12:00:00.123",
                                                    "18321 days 00:01:01.007",
                                                    "16798 days 23:02:02.42",
                                                    "15258 days 03:03:03.999",
                                                    "-70672 days +11:59:59.953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);

  using ns = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_ns{ns{1530705600123456789},
                                                                         ns{1582934461007008009},
                                                                         ns{1451430122421310209},
                                                                         ns{1318302183999777550},
                                                                         ns{-6106017600047047047}};
  results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S.%9f");
  cudf::test::strings_column_wrapper expected_ns_9f{"17716 days 12:00:00.123456789",
                                                 "18321 days 00:01:01.007008009",
                                                 "16798 days 23:02:02.421310209",
                                                 "15258 days 03:03:03.999777550",
                                                 "-70672 days +11:59:59.952952953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_9f);

  results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S.%6f");
  cudf::test::strings_column_wrapper expected_ns_6f{"17716 days 12:00:00.123456",
                                                    "18321 days 00:01:01.007008",
                                                    "16798 days 23:02:02.421310",
                                                    "15258 days 03:03:03.999777",
                                                    "-70672 days +11:59:59.952952"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_6f);

  results = cudf::strings::from_durations(durations_ns, "%d days %+%H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ns{"17716 days 12:00:00.123456789",
                                                 "18321 days 00:01:01.007008009",
                                                 "16798 days 23:02:02.421310209",
                                                 "15258 days 03:03:03.99977755",
                                                 "-70672 days +11:59:59.952952953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);
}

TEST_F(StringsDurationsTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::type_id::DURATION_SECONDS}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::from_durations(zero_size_column);
  cudf::test::expect_strings_empty(results->view());

  /* TODO
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  results = cudf::strings::to_durations(cudf::strings_column_view(zero_size_strings_column),
                                         cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
                                         "%Y");
  EXPECT_EQ(0, results->size());*/
}

TEST_F(StringsDurationsTest, Errors)
{
  // TODO to_durations
  cudf::test::fixed_width_column_wrapper<int64_t> invalid_durations{1530705600};
  EXPECT_THROW(cudf::strings::from_durations(invalid_durations), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations{cudf::duration_s{1530705600}};
  EXPECT_THROW(cudf::strings::from_durations(durations, ""), cudf::logic_error);
}

TEST_F(StringsDurationsTest, duration_s)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},          // 0 days 00:00:00
    T{1L},          // 0 days 00:00:01
    T{118800L},     // 1 days 09:00:00
    T{31568404L},   // 365 days 09:00:04
    T{-118800L},    // -2 days +15:00:00
    T{-31568404L},  // -366 days +14:59:56
  };
  auto results = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "0 days 00:00:01",
                                              "1 days 09:00:00",
                                              "365 days 09:00:04",
                                              "-2 days +15:00:00",
                                              "-366 days +14:59:56"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDurationsTest, duration_D)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},            // 0 days 00:00:00
    T{1L},            // 1 days 00:00:00
    T{-1L},           // -1 days +00:00:00
    T{800L},          // 800 days 00:00:00
    T{-800L},         // -800 days +00:00:00
    T{2147483647L},   //  2147483647 days 00:00:00
    T{-2147483648L},  // -2147483648 days +00:00:00
  };
  auto results = cudf::strings::from_durations(durations, "%d days %+%H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "1 days 00:00:00",
                                              "-1 days +00:00:00",
                                              "800 days 00:00:00",
                                              "-800 days +00:00:00",
                                              "2147483647 days 00:00:00",
                                              "-2147483648 days +00:00:00"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
